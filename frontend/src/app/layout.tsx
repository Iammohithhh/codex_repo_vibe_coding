import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Paper2Product 2.0 — AI Research OS',
  description: 'Transform research papers into production-ready assets',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
