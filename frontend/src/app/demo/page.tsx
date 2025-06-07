// frontend/src/app/demo/page.tsx
import React from 'react';
import DemoChatWindow from '@/components/DemoChatWindow'; // Adjusted import path

export default function DemoPage() {
  return (
    <section className="w-full flex flex-col items-center justify-start py-4 md:py-8">
      {/*
        The main Layout component (from src/app/layout.tsx)
        provides header, footer, and overall page structure.
        This DemoPage component will render its content within that Layout.
      */}
      <DemoChatWindow />
    </section>
  );
}
