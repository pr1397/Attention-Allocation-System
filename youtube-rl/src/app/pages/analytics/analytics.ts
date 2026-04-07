import { ChangeDetectorRef, Component, OnInit } from '@angular/core';
import { CommonModule, DecimalPipe } from '@angular/common';
import { AgentResult, AnalyticsData, RecommendationService } from '../../core/services/recommendation';

@Component({
  selector: 'app-analytics',
  standalone: true,
  imports: [CommonModule, DecimalPipe],
  templateUrl: './analytics.html',
  styleUrls: ['./analytics.css']
})
export class AnalyticsComponent implements OnInit {

  data: AnalyticsData | null = null;
  loading  = false;
  error    = '';
  selected = 'easy';   // active task tab

  tasks = ['easy', 'medium', 'hard'];
  agents = [
    { key: 'greedy',     label: 'Greedy',     color: '#534AB7', bg: '#EEEDFE' },
    { key: 'q_learning', label: 'Q-Learning',  color: '#BA7517', bg: '#FAEEDA' },
  ];

  constructor(private recService: RecommendationService, private cdr: ChangeDetectorRef) {}

  ngOnInit() {
    this.loadAnalytics();
  }

  loadAnalytics() {
    this.loading = true;
    this.error   = '';
   this.recService.getAnalytics().subscribe({
  next: (d: any) => {
    console.log('Analytics response:', d);
    this.data = d;
    this.loading = false;
    this.cdr.detectChanges(); // force UI update
  },
  error: (err) => {
    console.error('Analytics error:', err);
    this.error = 'Could not reach backend. Make sure Python server is running on port 7860.';
    this.loading = false;
  }
});

  }

  getResult(task: string, agent: string): AgentResult | null {
    if (!this.data) return null;
    return (this.data as any)[task]?.[agent] ?? null;
  }

  getScore(task: string, agent: string): number {
    return this.getResult(task, agent)?.score ?? 0;
  }

  getBarWidth(task: string, agent: string): string {
    return `${this.getScore(task, agent) * 100}%`;
  }

  getBestAgent(task: string): string {
    if (!this.data) return '';
    const scores = this.agents.map(a => ({
      label: a.label,
      score: this.getScore(task, a.key)
    }));
    return scores.sort((a, b) => b.score - a.score)[0]?.label ?? '';
  }

  getSteps(task: string, agent: string) {
    return this.getResult(task, agent)?.steps ?? [];
  }

  maxCumulative(task: string): number {
    let max = 0;
    for (const a of this.agents) {
      const steps = this.getSteps(task, a.key);
      if (steps.length) max = Math.max(max, steps[steps.length-1].cumulative);
    }
    return max || 1;
  }

  rewardBarHeight(task: string, agent: string, step: any): string {
    const max = this.maxCumulative(task);
    return `${(step.cumulative / max) * 100}%`;
  }
}