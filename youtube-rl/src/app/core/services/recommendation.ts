import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface Video {
  id: number;
  title: string;
  thumbnail: string;
  channel: string;
  category: string;
  duration: string;
  views: string;
  quality: number;
  length: number;
}

export interface AnalyticsData {
  easy:   TaskResult;
  medium: TaskResult;
  hard:   TaskResult;
}

export interface TaskResult {
  greedy:     AgentResult;
  q_learning: AgentResult;
  dqn:        AgentResult;
}

export interface AgentResult {
  steps:        Step[];
  total_reward: number;
  score:        number;
  steps_taken:  number;
}

export interface Step {
  step:       number;
  item_id:    number;
  reward:     number;
  fatigue:    number;
  cumulative: number;
}

@Injectable({ providedIn: 'root' })
export class RecommendationService {

  private baseUrl = 'http://localhost:7860';

  constructor(private http: HttpClient) {}

  getFeed(): Observable<Video[]> {
    return this.http.get<Video[]>(`${this.baseUrl}/recommend-feed`);
  }

  sendFeedback(videoId: number, action: string, watchTime = 0): Observable<any> {
    return this.http.post(`${this.baseUrl}/feedback`, {
      video_id:   videoId,
      action:     action,
      watch_time: watchTime,
    });
  }

  getAnalytics(): Observable<AnalyticsData> {
    return this.http.get<AnalyticsData>(`${this.baseUrl}/analytics`);
  }

  resetSession(): Observable<any> {
    return this.http.post(`${this.baseUrl}/reset`, {});
  }
}