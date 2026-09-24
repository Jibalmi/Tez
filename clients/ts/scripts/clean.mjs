// Remove dist/ before a build (no rimraf needed).
import { rmSync } from "node:fs";

rmSync(new URL("../dist", import.meta.url), { recursive: true, force: true });
