// results.json is large and its shape is owned by CONTRACT.md, not by the TypeScript
// compiler. We deliberately keep `resolveJsonModule` off so tsc never has to infer a
// multi-megabyte literal type, and instead hand the import to the runtime validator in
// lib/results.ts, which is the single place that knows the contract.
declare module "*.json" {
  const value: unknown;
  export default value;
}
