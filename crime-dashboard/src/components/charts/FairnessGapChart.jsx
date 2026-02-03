import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { groupColors } from '../../utils/colorPalette';

const FairnessGapChart = ({ data }) => {
  const svgRef = useRef();
  const containerRef = useRef();

  useEffect(() => {
    if (!data || data.length === 0) return;

    // Clear previous chart
    d3.select(svgRef.current).selectAll('*').remove();

    // Dimensions
    const containerWidth = containerRef.current.offsetWidth;
    const margin = { top: 20, right: 120, bottom: 60, left: 60 };
    const width = containerWidth - margin.left - margin.right;
    const height = 400 - margin.top - margin.bottom;

    // Create SVG
    const svg = d3.select(svgRef.current)
      .attr('width', containerWidth)
      .attr('height', 400)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Get all groups
    const allGroups = Array.from(new Set(data.flatMap(d => d.map(g => g.group))));
    
    // Get all model names (since we're dealing with a nested structure)
    const modelNames = Array.from({ length: data.length }, (_, i) => `Model ${i + 1}`);

    // Scales
    const x = d3.scaleBand()
      .domain(modelNames)
      .range([0, width])
      .padding(0.2);

    const y = d3.scaleLinear()
      .domain([0, d3.max(data.flat(), d => d.mae) * 1.1])
      .range([height, 0]);

    const xGroup = d3.scaleBand()
      .domain(allGroups)
      .range([0, x.bandwidth()])
      .padding(0.05);

    // X Axis
    svg.append('g')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(x))
      .style('font-size', '12px');

    // Y Axis
    svg.append('g')
      .call(d3.axisLeft(y))
      .style('font-size', '12px');

    // Y Axis Label
    svg.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('y', 0 - margin.left)
      .attr('x', 0 - (height / 2))
      .attr('dy', '1em')
      .style('text-anchor', 'middle')
      .style('font-size', '14px')
      .style('font-weight', 'bold')
      .text('MAE');

    // Grouped bars
    svg.selectAll('.model-group')
      .data(data)
      .enter()
      .append('g')
      .attr('class', 'model-group')
      .attr('transform', (d, i) => `translate(${x(`Model ${i + 1}`)},0)`)
      .selectAll('rect')
      .data((d, modelIdx) => d.map(group => ({...group, modelIdx})))  // Add model index to each group
      .enter()
      .append('rect')
      .attr('x', d => xGroup(d.group))
      .attr('y', height)
      .attr('width', xGroup.bandwidth())
      .attr('height', 0)
      .attr('fill', d => groupColors[d.group] || '#3b82f6')
      .attr('opacity', 0.8)
      .on('mouseover', function(event, d) {
        d3.select(this).attr('opacity', 1);
        
        const tooltip = d3.select('body')
          .append('div')
          .attr('class', 'tooltip')
          .style('position', 'absolute')
          .style('background', 'rgba(0, 0, 0, 0.8)')
          .style('color', 'white')
          .style('padding', '8px')
          .style('border-radius', '4px')
          .style('font-size', '12px')
          .style('pointer-events', 'none')
          .style('z-index', '1000')
          .html(`
            <strong>${d.group}</strong><br/>
            MAE: ${d.mae.toFixed(2)}<br/>
            R²: ${d.r2.toFixed(4)}<br/>
            Count: ${d.count}
          `);

        tooltip
          .style('left', (event.pageX + 10) + 'px')
          .style('top', (event.pageY - 20) + 'px');
      })
      .on('mouseout', function() {
        d3.select(this).attr('opacity', 0.8);
        d3.selectAll('.tooltip').remove();
      })
      .transition()
      .duration(800)
      .attr('y', d => y(d.mae))
      .attr('height', d => height - y(d.mae));

    // Add model labels on x-axis
    svg.selectAll('.model-label')
      .data(modelNames)
      .enter()
      .append('text')
      .attr('class', 'model-label')
      .attr('x', (d, i) => x(d) + x.bandwidth() / 2)
      .attr('y', height + 35)
      .attr('text-anchor', 'middle')
      .style('font-size', '11px')
      .text((d, i) => data[i]?.[0]?.group ? `Model ${i + 1}` : d);

    // Legend
    const legend = svg.append('g')
      .attr('class', 'legend')
      .attr('transform', `translate(${width + 20}, 0)`);

    allGroups.forEach((group, i) => {
      const legendRow = legend.append('g')
        .attr('transform', `translate(0, ${i * 25})`);

      legendRow.append('rect')
        .attr('width', 15)
        .attr('height', 15)
        .attr('fill', groupColors[group]);

      legendRow.append('text')
        .attr('x', 20)
        .attr('y', 12)
        .style('font-size', '12px')
        .text(group);
    });

  }, [data]);

  return (
    <div ref={containerRef} className="w-full">
      <svg ref={svgRef}></svg>
    </div>
  );
};

export default FairnessGapChart;