import { AfterViewInit, Component, ElementRef, Inject, PLATFORM_ID, ViewChild } from '@angular/core';
import { RecommendationService } from '../core/services/recommendation';
import { Chart } from 'chart.js';
import { CommonModule, DecimalPipe, isPlatformBrowser } from '@angular/common';

@Component({
  selector: 'app-analytics',
  imports: [DecimalPipe, CommonModule],
  templateUrl: './analytics.html',
  styleUrl: './analytics.css',
})

export class Analytics implements AfterViewInit{
  @ViewChild('chartCanvas') chartRef!: ElementRef<HTMLCanvasElement>;

  data: any[] = [];

  accuracy = 0;
  avgReward = 0;
  skipRate = 0;

  chart: any;

  constructor(
    private recService: RecommendationService,
    @Inject(PLATFORM_ID) private platformId: Object
  ) {}

  // Entry point
  ngAfterViewInit() {
    this.loadAnalytics();
  }

  //  STEP 2 — Fetch data
  loadAnalytics() {
    this.recService.getAnalytics().subscribe(res => {
      console.log("ANALYTICS DATA:", res);

      this.data = res || [];

      if (this.data.length) {
        this.calculateMetrics();

        // wait for DOM
        setTimeout(() => {
          this.renderChart();
        }, 0);
      }
    });
  }

  //  STEP 3 — Metrics
  calculateMetrics() {
    const total = this.data.length;

    if (!total) return;

    const positive = this.data.filter(d => d.reward > 0).length;
    const skips = this.data.filter(d => d.action === 'skip').length;

    const totalReward = this.data.reduce(
      (sum, d) => sum + (d.reward || 0),
      0
    );

    this.accuracy = positive / total;
    this.avgReward = totalReward / total;
    this.skipRate = skips / total;
  }

  //  STEP 4 — Chart (SSR SAFE)
  renderChart() {

    //  Prevent SSR crash
    if (!isPlatformBrowser(this.platformId)) return;

    if (!this.chartRef) return;

    const ctx = this.chartRef.nativeElement.getContext('2d');
    if (!ctx) return;

    // destroy old chart if exists
    if (this.chart) {
      this.chart.destroy();
    }

    this.chart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: this.data.map((_, i) => i + 1),
        datasets: [
          {
            label: 'Reward Over Time',
            data: this.data.map(d => d.reward || 0)
          }
        ]
      },
      options: {
        responsive: true
      }
    });
  }
}

