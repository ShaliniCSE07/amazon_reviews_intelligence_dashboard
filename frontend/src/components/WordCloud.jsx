import React, { useEffect, useRef } from 'react';

const WordCloud = ({ words = [], theme = 'light' }) => {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Reset and resize canvas
    const width = canvas.offsetWidth;
    const height = canvas.offsetHeight || 250;
    canvas.width = width;
    canvas.height = height;

    ctx.clearRect(0, 0, width, height);

    if (!words || words.length === 0) {
      ctx.font = '14px Inter, sans-serif';
      ctx.fillStyle = theme === 'dark' ? '#94a3b8' : '#64748b';
      ctx.textAlign = 'center';
      ctx.fillText('No keyword data available', width / 2, height / 2);
      return;
    }

    // Sort words by value descending
    const sortedWords = [...words]
      .filter(w => w.text && w.value)
      .sort((a, b) => b.value - a.value)
      .slice(0, 30); // Limit to top 30 words for clean layout

    if (sortedWords.length === 0) return;

    const maxVal = sortedWords[0].value;
    const minVal = sortedWords[sortedWords.length - 1].value;

    // Font size scaling (from 12px to 38px)
    const getFontSize = (val) => {
      if (maxVal === minVal) return 20;
      const pct = (val - minVal) / (maxVal - minVal);
      return Math.round(12 + pct * 26);
    };

    // Theme color schemes
    const lightColors = ['#6366f1', '#8b5cf6', '#3b82f6', '#ec4899', '#f43f5e', '#10b981', '#f59e0b'];
    const darkColors = ['#a78bfa', '#c084fc', '#60a5fa', '#f472b6', '#fb7185', '#34d399', '#fbbf24'];
    const colors = theme === 'dark' ? darkColors : lightColors;

    const placedBoxes = [];

    const intersects = (box1, box2) => {
      return !(
        box1.x + box1.w < box2.x ||
        box2.x + box2.w < box1.x ||
        box1.y + box1.h < box2.y ||
        box2.y + box2.h < box1.y
      );
    };

    const isOverlap = (box) => {
      for (const placed of placedBoxes) {
        if (intersects(box, placed)) return true;
      }
      return false;
    };

    // Draw each word
    sortedWords.forEach((wordObj, index) => {
      const fontSize = getFontSize(wordObj.value);
      ctx.font = `bold ${fontSize}px Outfit, Inter, sans-serif`;
      
      const textWidth = ctx.measureText(wordObj.text).width;
      const textHeight = fontSize;

      let angle = 0;
      let radius = 0;
      let x = width / 2;
      let y = height / 2;
      
      let placed = false;
      const spiralStep = 3;
      const maxAttempts = 300;
      let attempt = 0;

      // Spiral search to place the word
      while (attempt < maxAttempts && !placed) {
        // Archimedean Spiral coordinates
        const theta = angle;
        radius = spiralStep * (theta / (2 * Math.PI));
        
        x = width / 2 + radius * Math.cos(theta) - textWidth / 2;
        y = height / 2 + radius * Math.sin(theta) - textHeight / 2;

        const box = {
          x: x - 4, // add padding
          y: y - textHeight,
          w: textWidth + 8,
          h: textHeight + 6,
          text: wordObj.text,
          fontSize,
          color: colors[index % colors.length]
        };

        // Bounds checks
        if (box.x >= 0 && box.x + box.w <= width && box.y >= 0 && box.y + box.h <= height) {
          if (!isOverlap(box)) {
            placedBoxes.push(box);
            placed = true;
          }
        }

        angle += 0.15;
        attempt++;
      }

      // Draw word if placed, or fallback center placement if first word
      if (!placed && index === 0) {
        // Place first word in middle regardless
        const box = {
          x: width / 2 - textWidth / 2,
          y: height / 2 - textHeight / 2,
          w: textWidth,
          h: textHeight,
          text: wordObj.text,
          fontSize,
          color: colors[0]
        };
        placedBoxes.push(box);
      }
    });

    // Actually render the placed boxes
    placedBoxes.forEach(box => {
      ctx.font = `bold ${box.fontSize}px Outfit, Inter, sans-serif`;
      ctx.fillStyle = box.color;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'alphabetic';
      ctx.fillText(box.text, box.x + 4, box.y + box.fontSize);
    });

  }, [words, theme]);

  return (
    <div className="w-full h-full relative flex items-center justify-center">
      <canvas ref={canvasRef} className="w-full h-full min-h-[200px]" />
    </div>
  );
};

export default WordCloud;
