/**
 * Gemini 3 compatibility for Google's OpenAI-compatible endpoint.
 *
 * Gemini 3 returns a "thought signature" with each tool call
 * (`tool_calls[].extra_content.google.thought_signature`) and rejects the next
 * request with HTTP 400 ("Function call is missing a thought_signature")
 * unless the signature is sent back on that tool call. The OpenAI-compatible
 * chat adapter does not carry `extra_content`, so every turn that uses a tool
 * fails after the tool runs. This fetch wrapper remembers the signatures it
 * sees in responses and restores them on the matching tool calls of later
 * requests. Tool calls it has never seen (for example history reloaded after a
 * restart) get Google's documented placeholder, which the API accepts.
 */

const GOOGLE_HOST = 'generativelanguage.googleapis.com';
const PLACEHOLDER_SIGNATURE = 'skip_thought_signature_validator';
const MAX_REMEMBERED = 5000;

const remembered = new Map<string, string>();

export function isGoogleOpenAIEndpoint(baseUrl?: string): boolean {
  if (!baseUrl) return false;
  try {
    return new URL(baseUrl).hostname === GOOGLE_HOST;
  } catch {
    return false;
  }
}

function field(value: unknown, key: string): unknown {
  return typeof value === 'object' && value !== null
    ? (value as Record<string, unknown>)[key]
    : undefined;
}

function signatureOf(call: unknown): string | undefined {
  const signature = field(
    field(field(call, 'extra_content'), 'google'),
    'thought_signature',
  );
  return typeof signature === 'string' && signature ? signature : undefined;
}

function remember(calls: unknown): void {
  if (!Array.isArray(calls)) return;
  for (const call of calls) {
    const id = field(call, 'id');
    const signature = signatureOf(call);
    if (typeof id !== 'string' || !id || !signature) continue;
    remembered.delete(id);
    remembered.set(id, signature);
    if (remembered.size > MAX_REMEMBERED) {
      const oldest = remembered.keys().next().value;
      if (oldest !== undefined) remembered.delete(oldest);
    }
  }
}

function rememberFromChunk(chunk: unknown): void {
  const choices = field(chunk, 'choices');
  if (!Array.isArray(choices)) return;
  for (const choice of choices) {
    remember(field(field(choice, 'delta'), 'tool_calls'));
    remember(field(field(choice, 'message'), 'tool_calls'));
  }
}

async function rememberFromResponse(response: Response): Promise<void> {
  const text = await response.text();
  if ((response.headers.get('content-type') ?? '').includes('event-stream')) {
    for (const line of text.split('\n')) {
      const data = line.trim();
      if (!data.startsWith('data:') || data.includes('[DONE]')) continue;
      try {
        rememberFromChunk(JSON.parse(data.slice(5)));
      } catch {
        // A partial or non-JSON event carries no signature.
      }
    }
    return;
  }
  try {
    rememberFromChunk(JSON.parse(text));
  } catch {
    // Not a completion body.
  }
}

/** Adds a thought signature to every assistant tool call that lacks one. */
export function restoreThoughtSignatures(body: unknown): number {
  const messages = field(body, 'messages');
  if (!Array.isArray(messages)) return 0;
  let restored = 0;
  for (const message of messages) {
    if (field(message, 'role') !== 'assistant') continue;
    const calls = field(message, 'tool_calls');
    if (!Array.isArray(calls)) continue;
    for (const call of calls) {
      if (typeof call !== 'object' || call === null || signatureOf(call))
        continue;
      const id = field(call, 'id');
      const target = call as Record<string, unknown>;
      const extra = field(target, 'extra_content');
      const google = field(extra, 'google');
      target.extra_content = {
        ...(typeof extra === 'object' && extra !== null ? extra : {}),
        google: {
          ...(typeof google === 'object' && google !== null ? google : {}),
          thought_signature:
            (typeof id === 'string' ? remembered.get(id) : undefined) ??
            PLACEHOLDER_SIGNATURE,
        },
      };
      restored += 1;
    }
  }
  return restored;
}

function urlOf(input: Parameters<typeof fetch>[0]): string {
  if (typeof input === 'string') return input;
  return input instanceof URL ? input.href : input.url;
}

/** Wraps fetch for the OpenAI client so Gemini tool-call round trips succeed. */
export function geminiToolCallFetch(
  inner: typeof fetch = (input, init) => globalThis.fetch(input, init),
): typeof fetch {
  return async (input, init) => {
    let request = init;
    if (
      typeof init?.body === 'string' &&
      urlOf(input).split('?')[0].endsWith('/chat/completions')
    ) {
      try {
        const body: unknown = JSON.parse(init.body);
        if (restoreThoughtSignatures(body) > 0)
          request = { ...init, body: JSON.stringify(body) };
      } catch {
        // Not JSON: send it unchanged.
      }
    }
    const response = await inner(input, request);
    if (response.ok && response.body)
      void rememberFromResponse(response.clone()).catch(() => undefined);
    return response;
  };
}
