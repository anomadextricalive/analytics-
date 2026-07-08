/**
 * MongoDB Atlas – Node.js quickstart (activity feed)
 *
 * Install: npm install mongodb dotenv
 * Run:     MONGODB_URI="mongodb+srv://..." node mongodbExample.js
 *          — or put MONGODB_URI in a .env file and just run: node mongodbExample.js
 */

"use strict";

const { MongoClient, ObjectId } = require("mongodb");

// dotenv is optional — if the package isn't installed we simply skip it.
// This lets MONGODB_URI come from the shell environment instead.
try { require("dotenv").config(); } catch (_) {}

// ── Connection ────────────────────────────────────────────────────────────────

// Never hard-code credentials. Pull from the environment so secrets stay out
// of source control.
const uri = process.env.MONGODB_URI;
if (!uri) {
  console.error(
    "Error: MONGODB_URI environment variable is not set.\n" +
    "Set it in your shell or add it to a .env file."
  );
  process.exit(1);
}

// A single MongoClient instance is reused for every operation — creating a new
// client per query is expensive and exhausts connection pools quickly.
const client = new MongoClient(uri);

// ── Sample data ───────────────────────────────────────────────────────────────

// Ten realistic activity-feed events spread across the last 10 days.
// Varying timestamps let us demonstrate time-based sorting.
function buildActivities() {
  const now = Date.now();
  const DAY = 24 * 60 * 60 * 1000;

  const events = [
    { user: "alice",   action: "posted",   target: "match recap vs Australia" },
    { user: "bob",     action: "liked",    target: "player rating: Kohli" },
    { user: "carol",   action: "commented",target: "pitch report: Wankhede" },
    { user: "dave",    action: "shared",   target: "H2H comparison: Rohit vs Babar" },
    { user: "eve",     action: "followed", target: "user: frank" },
    { user: "frank",   action: "posted",   target: "IPL 2025 predicted XI" },
    { user: "grace",   action: "liked",    target: "match recap vs England" },
    { user: "heidi",   action: "commented",target: "venue stats: Eden Gardens" },
    { user: "ivan",    action: "shared",   target: "bowler matchup analysis" },
    { user: "judy",    action: "posted",   target: "T20 WC squad predictions" },
  ];

  // Spread events 1 day apart so sorting by createdAt gives a meaningful order.
  return events.map((e, i) => ({
    ...e,
    createdAt: new Date(now - i * DAY), // newest first in the array
    likes: Math.floor(Math.random() * 200),
    comments: Math.floor(Math.random() * 50),
  }));
}

// ── Main ──────────────────────────────────────────────────────────────────────

async function main() {
  console.log("→ Connecting to MongoDB Atlas…");
  await client.connect();
  console.log("✓ Connected.\n");

  // Namespace: database → collection.  Neither needs to exist beforehand —
  // MongoDB creates them on first write.
  const db         = client.db("cricket_analytics");
  const activities = db.collection("activity_feed");

  // ── Insert ────────────────────────────────────────────────────────────────

  console.log("→ Inserting 10 activity documents…");
  const docs   = buildActivities();
  const result = await activities.insertMany(docs);
  console.log(`✓ Inserted ${result.insertedCount} documents.\n`);

  // ── Read: 5 most recent ────────────────────────────────────────────────────

  // sort({ createdAt: -1 }) = descending (newest first).
  // limit(5) fetches only what we need — never pull more than the UI requires.
  console.log("── 5 most recent activities ─────────────────────────────────");
  const recent = await activities
    .find()
    .sort({ createdAt: -1 })
    .limit(5)
    .toArray();

  recent.forEach((doc, i) => {
    console.log(
      `${i + 1}. [${doc.createdAt.toISOString().slice(0, 10)}] ` +
      `${doc.user} ${doc.action}: "${doc.target}" ` +
      `(${doc.likes} likes, _id: ${doc._id})`
    );
  });
  console.log();

  // ── Read: single document by _id ──────────────────────────────────────────

  // _id values are ObjectId instances, not plain strings — wrap accordingly.
  const firstId  = result.insertedIds[0];
  console.log(`── Fetching single document by _id: ${firstId} ─────────────`);
  const single = await activities.findOne({ _id: firstId });
  console.log(JSON.stringify(single, null, 2));
  console.log();
}

// ── Run ───────────────────────────────────────────────────────────────────────

main()
  .catch((err) => {
    // Surface the real error rather than swallowing it — beginners need to see
    // what went wrong (auth failure, network, wrong URI format, etc.).
    console.error("✗ Error:", err.message);
    process.exit(1);
  })
  .finally(async () => {
    // Always close the connection, even if an error occurred.  Leaving it open
    // keeps the Node.js process alive indefinitely.
    await client.close();
    console.log("✓ Connection closed.");
  });
