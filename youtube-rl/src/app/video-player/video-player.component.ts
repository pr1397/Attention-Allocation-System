import { Component } from '@angular/core';
import { ActivatedRoute } from '@angular/router';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-video-player',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div style="padding:80px 24px;color:white;text-align:center">
      <h2>Video Player</h2>
      <p style="color:rgba(255,255,255,0.5)">Video ID: {{ videoId }}</p>
    </div>
  `
})
export class VideoPlayerComponent {
  videoId: string | null = null;
  constructor(private route: ActivatedRoute) {
    this.videoId = this.route.snapshot.paramMap.get('id');
  }
}