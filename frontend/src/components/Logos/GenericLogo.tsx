import { useEffect, useState } from 'react';
import MetallicPaint from '../MetallicPaint';

export default function GenericLogo({ className = '' }: { className?: string }) {
  const [imageData, setImageData] = useState<ImageData | null>(null);

  useEffect(() => {
    // Create a simple generic logo with text
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = 800;
    canvas.height = 800;

    // Clear background
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw a circle with "UC" text (Urban Climate)
    ctx.fillStyle = 'white';
    ctx.beginPath();
    ctx.arc(400, 400, 350, 0, Math.PI * 2);
    ctx.fill();

    // Draw text
    ctx.fillStyle = 'black';
    ctx.font = 'bold 280px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('UC', 400, 400);

    // Get image data
    const data = ctx.getImageData(0, 0, canvas.width, canvas.height);
    setImageData(data);
  }, []);

  if (!imageData) {
    return <div className={`${className} animate-pulse bg-slate-700 rounded-full`} />;
  }

  return (
    <div className={className}>
      <MetallicPaint
        imageData={imageData}
        params={{
          patternScale: 2,
          refraction: 0.015,
          edge: 1,
          patternBlur: 0.005,
          liquid: 0.07,
          speed: 0.3,
        }}
      />
    </div>
  );
}
