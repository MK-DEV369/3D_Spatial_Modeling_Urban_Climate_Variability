import { useEffect, useState } from 'react';
import MetallicPaint from '../MetallicPaint';

export default function RVCELogo({ className = '' }: { className?: string }) {
  const [imageData, setImageData] = useState<ImageData | null>(null);

  useEffect(() => {
    // Create RVCE logo with text
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = 800;
    canvas.height = 800;

    // Clear background
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw a shield shape
    ctx.fillStyle = 'white';
    ctx.beginPath();
    ctx.moveTo(400, 50);
    ctx.lineTo(700, 250);
    ctx.lineTo(700, 600);
    ctx.quadraticCurveTo(400, 750, 100, 600);
    ctx.lineTo(100, 250);
    ctx.closePath();
    ctx.fill();

    // Draw "RVCE" text
    ctx.fillStyle = 'black';
    ctx.font = 'bold 180px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('RVCE', 400, 350);

    // Draw subtitle
    ctx.font = 'bold 60px Arial';
    ctx.fillText('Bangalore', 400, 520);

    // Get image data
    const data = ctx.getImageData(0, 0, canvas.width, canvas.height);
    setImageData(data);
  }, []);

  if (!imageData) {
    return <div className={`${className} animate-pulse bg-slate-700`} />;
  }

  return (
    <div className={className}>
      <MetallicPaint
        imageData={imageData}
        params={{
          patternScale: 2.5,
          refraction: 0.02,
          edge: 1.2,
          patternBlur: 0.003,
          liquid: 0.08,
          speed: 0.25,
        }}
      />
    </div>
  );
}
