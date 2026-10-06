import { expect, it, vi } from 'vitest';
import {
  geminiToolCallFetch,
  isGoogleOpenAIEndpoint,
  restoreThoughtSignatures,
} from '../src/server/gemini-compat.js';

const PLACEHOLDER = 'skip_thought_signature_validator';
const ENDPOINT =
  'https://generativelanguage.googleapis.com/v1beta/openai/chat/completions';

function toolCallStream(id: string, signature: string) {
  const chunk = {
    id: 'completion',
    object: 'chat.completion.chunk',
    created: 1,
    model: 'gemini-3',
    choices: [
      {
        index: 0,
        delta: {
          role: 'assistant',
          tool_calls: [
            {
              id,
              type: 'function',
              function: { name: 'lookup', arguments: '{}' },
              extra_content: { google: { thought_signature: signature } },
            },
          ],
        },
        finish_reason: null,
      },
    ],
  };
  const text = `data: ${JSON.stringify(chunk)}\n\ndata: [DONE]\n\n`;
  return new Response(text, {
    headers: { 'Content-Type': 'text/event-stream' },
  });
}

function history(id: string) {
  return {
    model: 'gemini-3',
    messages: [
      { role: 'user', content: 'Look it up.' },
      {
        role: 'assistant',
        content: null,
        tool_calls: [
          {
            id,
            type: 'function',
            function: { name: 'lookup', arguments: '{}' },
          },
        ],
      },
      { role: 'tool', tool_call_id: id, content: '{}' },
    ],
  };
}

function signatureSent(init: RequestInit | undefined) {
  const body = JSON.parse(String(init?.body));
  return body.messages[1].tool_calls[0].extra_content?.google
    ?.thought_signature;
}

it('recognizes only Google endpoints', () => {
  expect(
    isGoogleOpenAIEndpoint(
      'https://generativelanguage.googleapis.com/v1beta/openai/',
    ),
  ).toBe(true);
  expect(isGoogleOpenAIEndpoint('https://api.openai.com/v1')).toBe(false);
  expect(
    isGoogleOpenAIEndpoint(
      'https://generativelanguage.googleapis.com.evil.test/v1',
    ),
  ).toBe(false);
  expect(isGoogleOpenAIEndpoint('not a url')).toBe(false);
  expect(isGoogleOpenAIEndpoint(undefined)).toBe(false);
});

it('sends back the signature Gemini returned, and leaves the response readable', async () => {
  const inner = vi
    .fn<typeof fetch>()
    .mockResolvedValueOnce(toolCallStream('call-real', 'SIG-REAL'))
    .mockImplementation(async () => new Response('{}'));
  const wrapped = geminiToolCallFetch(inner);

  const first = await wrapped(ENDPOINT, {
    method: 'POST',
    body: JSON.stringify({ model: 'gemini-3', messages: [] }),
  });
  expect(await first.text()).toContain('SIG-REAL');
  await vi.waitFor(async () => {
    await wrapped(ENDPOINT, {
      method: 'POST',
      body: JSON.stringify(history('call-real')),
    });
    expect(signatureSent(inner.mock.lastCall?.[1])).toBe('SIG-REAL');
  });
});

it('uses the documented placeholder for tool calls it has not seen', async () => {
  const inner = vi.fn<typeof fetch>().mockResolvedValue(new Response('{}'));
  await geminiToolCallFetch(inner)(ENDPOINT, {
    method: 'POST',
    body: JSON.stringify(history('call-from-before-restart')),
  });
  expect(signatureSent(inner.mock.calls[0][1])).toBe(PLACEHOLDER);
});

it('keeps a signature that is already present', () => {
  const body = history('call-own');
  Object.assign(body.messages[1].tool_calls![0], {
    extra_content: { google: { thought_signature: 'MINE' } },
  });
  expect(restoreThoughtSignatures(body)).toBe(0);
  expect(JSON.stringify(body)).toContain('MINE');
  expect(JSON.stringify(body)).not.toContain(PLACEHOLDER);
});

it('does not touch other requests or non-JSON bodies', async () => {
  const inner = vi.fn<typeof fetch>().mockResolvedValue(new Response('{}'));
  const wrapped = geminiToolCallFetch(inner);
  const other = JSON.stringify(history('call-other'));
  await wrapped(
    'https://generativelanguage.googleapis.com/v1beta/openai/models',
    {
      method: 'POST',
      body: other,
    },
  );
  await wrapped(ENDPOINT, { method: 'POST', body: 'not json' });
  await wrapped(ENDPOINT, { method: 'GET' });
  expect(inner.mock.calls[0][1]?.body).toBe(other);
  expect(inner.mock.calls[1][1]?.body).toBe('not json');
  expect(inner.mock.calls[2][1]?.body).toBeUndefined();
});
