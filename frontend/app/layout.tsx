import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Colorectal Cancer Diagnosis",
  description: "Academic colorectal histopathology support system",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="es" suppressHydrationWarning><body>{children}</body></html>;
}
