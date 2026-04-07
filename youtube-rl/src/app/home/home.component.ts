import { ChangeDetectorRef, Component, HostListener } from '@angular/core';
import { CommonModule } from '@angular/common';
import { VideoCardComponent } from '../video-card/video-card.component';
import { RecommendationService } from '../core/services/recommendation';

@Component({
  selector: 'app-home',
  standalone: true,
  imports: [CommonModule, VideoCardComponent],
  templateUrl: './home.component.html',
  styleUrl: './home.component.css'
})
export class HomeComponent {

  videos: any[] = [];
  loading = false;// number of placeholders

  constructor(private recService: RecommendationService, private cdr: ChangeDetectorRef) {}

  ngOnInit() {
    this.loadFeed();
  }

  loadFeed() {
    if (this.loading) return;

    this.loading = true;

    this.recService.getFeed().subscribe({
      next: (res) => {
         console.log("API RESPONSE:", res);
          console.log("TYPE:", typeof res);
          console.log("LENGTH:", res?.length);
        this.videos = [...this.videos, ...res];
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
    if ((window.innerHeight + window.scrollY) >= document.body.offsetHeight - 200) {
      this.loadFeed();
    }
  }
}