import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  // results.json is imported (not fetched) so the data is bundled and works identically
  // in a static export, on a Vercel serverless function, and in dev.
  experimental: {},
};

export default nextConfig;
