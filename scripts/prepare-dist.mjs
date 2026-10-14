import { cpSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';

// Stage only installable artifacts; never copy source, credentials, or dependencies.
const destination = process.argv[2];
if (!destination) throw new Error('Usage: node scripts/prepare-dist.mjs <empty-directory>');
const target = resolve(destination);
if (target === process.cwd()) throw new Error('Destination must not be the source directory');
mkdirSync(target, { recursive: true });
const pkg = JSON.parse(readFileSync('package.json', 'utf8'));
delete pkg.scripts;
delete pkg.devDependencies;
writeFileSync(resolve(target, 'package.json'), JSON.stringify(pkg, null, 2) + '\n');
cpSync('dist', resolve(target, 'dist'), { recursive: true });
cpSync('README.md', resolve(target, 'README.md'));
