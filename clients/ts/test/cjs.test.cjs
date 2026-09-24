// The CommonJS build, required through the package's own exports map.
const assert = require("node:assert/strict");
const { test } = require("node:test");

const cjs = require("tez-client");

test("require() loads the CommonJS build, and it works", async () => {
  assert.equal(typeof cjs.TezClient, "function");
  assert.equal(require.resolve("tez-client").replace(/\\/g, "/").endsWith("/dist/cjs/index.js"), true);
  const { startMockServer } = await import("./mock-server.mjs");
  const mock = await startMockServer();
  try {
    const client = new cjs.TezClient({ baseUrl: mock.url });
    const res = await client.decide("hello", { questions: { q: { type: "noul", instructions: "Is it?" } } });
    assert.deepEqual(res.answers.q, { type: "noul", noul: 0.8 });
    const err = await client.schema("nope").catch((e) => e);
    assert.ok(err instanceof cjs.TezError);
    assert.equal(err.status, 404);
  } finally {
    await mock.close();
  }
});

test("the ES module build is a separate file", async () => {
  const esm = await import("tez-client");
  assert.notEqual(esm.TezClient, cjs.TezClient);
  assert.equal(esm.DEFAULT_BASE_URL, cjs.DEFAULT_BASE_URL);
});
