import { 
  Component, Input, Output, EventEmitter, 
  OnChanges, ElementRef, AfterViewInit, Inject, PLATFORM_ID 
} from '@angular/core';
import { CommonModule, isPlatformBrowser } from '@angular/common';
import { Video } from '../core/services/recommendation';

@Component({
  selector: 'app-video-card',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './video-card.component.html',
  styleUrl: './video-card.component.css'
})
export class VideoCardComponent implements OnChanges, AfterViewInit {

  @Input()  video!: Video;
  @Input()  isActive = false;

  @Output() action = new EventEmitter<{ video: Video; action: string; watchTime?: number }>();
  @Output() visible = new EventEmitter<void>();

  liked = false;
  skipped = false;
  startTime: number | null = null;
  private observer!: IntersectionObserver;

  constructor(
    private el: ElementRef,
    @Inject(PLATFORM_ID) private platformId: Object
  ) {}

  ngAfterViewInit() {

    // ✅ IMPORTANT FIX
    if (!isPlatformBrowser(this.platformId)) return;

    this.observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          this.visible.emit();
          this.startTime = Date.now();
        } else if (this.startTime) {
          const watchTime = (Date.now() - this.startTime) / 1000;

          if (watchTime > 3) {
            this.action.emit({
              video: this.video,
              action: 'watch',
              watchTime
            });
          }

          this.startTime = null;
        }
      },
      { threshold: 0.7 }
    );

    this.observer.observe(this.el.nativeElement);
  }

  ngOnChanges() {
    if (this.isActive && this.startTime === null) {
      this.startTime = Date.now();
    }
  }

  onLike() {
    this.liked = !this.liked;
    this.action.emit({ video: this.video, action: 'click' });
  }

  onSkip() {
    this.skipped = true;
    this.action.emit({ video: this.video, action: 'skip' });
  }

  get qualityStars(): string {
    const stars = Math.round(this.video.quality * 5);
    return '★'.repeat(stars) + '☆'.repeat(5 - stars);
  }

  get rewardColor(): string {
    if (this.video.quality > 0.8) return '#4ade80';
    if (this.video.quality > 0.6) return '#fb923c';
    return '#94a3b8';
  }
}