import { ChangeDetectorRef, Component, HostListener } from '@angular/core';
import { CommonModule } from '@angular/common';
import { VideoCardComponent } from '../video-card/video-card.component';
import { RecommendationService } from '../core/services/recommendation';
import { routes } from '../app.routes';
import { provideRouter, RouterModule } from '@angular/router';

@Component({
  selector: 'app-home',
  standalone: true,
  imports: [CommonModule, VideoCardComponent, RouterModule],
  templateUrl: './home.component.html',
  styleUrl: './home.component.css'
})
export class HomeComponent {

  videos: any[] = [];
  loading = false;
  scrollTimeout: any;
  currentVideoIndex = 0;

  constructor(private recService: RecommendationService, private cdr: ChangeDetectorRef) {}

  ngOnInit() {
    this.loadFeed();
  }

  loadFeed() {
    if (this.loading) return;

    this.loading = true;

    this.recService.getFeed().subscribe({
      next: (res) => {
        this.videos.push(res); 
        this.loading = false;
        this.cdr.detectChanges();
      },
      error: () => {
        this.loading = false;
      }
    });
  }

  @HostListener('window:scroll', [])
  onScroll() {
    if (this.loading) return;

    const currentVideo = this.videos[this.currentVideoIndex];

    if (currentVideo) {
      this.recService.sendFeedback({
        video_id: currentVideo.id,
        action: 'skip', // default if user scrolls
        watch_time: 1   // assume low watch
      }).subscribe();
    }

    // increase fatigue
    this.recService.scroll().subscribe();

    // move to next video
    this.currentVideoIndex++;

    // load next recommendation
    this.loadFeed();
  }
}