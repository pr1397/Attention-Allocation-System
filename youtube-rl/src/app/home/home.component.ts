import { Component, OnInit, HostListener, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { VideoCardComponent } from '../video-card/video-card.component';
import { Video, RecommendationService } from '../core/services/recommendation';

@Component({
  selector: 'app-home',
  standalone: true,
  imports: [CommonModule, VideoCardComponent],
  templateUrl: './home.component.html',
  styleUrl: './home.component.css'
})
export class HomeComponent implements OnInit {

  videos: Video[]  = [];
  loading          = false;
  currentIndex     = 0;
  totalReward      = 0;
  sessionFatigue   = 0;

  constructor(private recService: RecommendationService, private cdr: ChangeDetectorRef) {}

  ngOnInit() {
    this.loadFeed();
  }

  loadFeed() {
    this.loading = true;
    this.recService.getFeed().subscribe({
      next: (videos) => {
        this.videos  = videos;
        this.loading = false;
        this.cdr.detectChanges(); // Ensure view updates after loading
      },
      error: () => {
        // Fallback demo videos if backend not running
        this.videos  = this.getDemoVideos();
        this.loading = false;
        this.cdr.detectChanges(); // Ensure view updates after loading
      }
    });
  }

  onVideoAction(event: { video: Video; action: string; watchTime?: number }) {
    this.recService.sendFeedback(event.video.id, event.action, event.watchTime || 0)
      .subscribe(res => {
        if (res.reward !== undefined) {
          this.totalReward    = Math.round((this.totalReward + res.reward) * 100) / 100;
          this.sessionFatigue = Math.round(Math.min(1.2, this.sessionFatigue + 0.1 * (event.video.length || 1)) * 100) / 100;
        }
        // Load more if session ended
        if (res.done) {
          this.totalReward    = 0;
          this.sessionFatigue = 0;
          this.loadFeed();
        }
      });
  }

  onIndexChange(index: number) {
    this.currentIndex = index;
  }

  getDemoVideos(): Video[] {
    const cats = ['Technology', 'Gaming', 'Education', 'Music', 'Fitness'];
    return Array.from({ length: 10 }, (_, i) => ({
      id:        i,
      title:     `${cats[i % 5]} — Demo Video ${i + 1}`,
      thumbnail: `https://picsum.photos/seed/${i + 200}/400/700`,
      channel:   `Demo Channel ${i % 4 + 1}`,
      category:  cats[i % 5],
      duration:  `${Math.floor(Math.random() * 12) + 1}:${String(Math.floor(Math.random() * 60)).padStart(2,'0')}`,
      views:     `${Math.floor(Math.random() * 900) + 100}K`,
      quality:   Math.round((Math.random() * 0.5 + 0.5) * 100) / 100,
      length:    (i % 5) + 1,
    }));
  }
}