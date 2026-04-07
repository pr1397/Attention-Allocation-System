import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RecommendationService } from '../core/services/recommendation';

@Component({
  selector: 'app-video-player',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './video-player.component.html'
})
export class VideoPlayerComponent {

  video: any;
  startTime = 0;

  constructor(private recService: RecommendationService) {
    this.video = history.state.video;
  }

  ngOnInit() {
    this.startTime = Date.now();
  }

  onEnd() {
    const watchTime = (Date.now() - this.startTime) / 1000;

    this.recService.sendFeedback({
      video_id: this.video.id,
      action: 'watch',
      watch_time: watchTime
    }).subscribe();
  }
}