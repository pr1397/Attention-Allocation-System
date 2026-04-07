import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class RecommendationService {

  private baseUrl = 'http://localhost:7860';

  constructor(private http: HttpClient) {}

  getFeed(): Observable<any[]> {
    return this.http.get<any[]>(`${this.baseUrl}/recommend-feed`);
  }

  sendFeedback(data: any) {
    return this.http.post(`${this.baseUrl}/feedback`, data);
  }
}