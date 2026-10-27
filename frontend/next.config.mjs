/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "http://127.0.0.1:8050/api/:path*",
      },
      {
        source: "/figures/:path*",
        destination: "http://127.0.0.1:8050/figures/:path*",
      },
    ];
  },
};

export default nextConfig;
