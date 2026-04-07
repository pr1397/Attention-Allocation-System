import { Routes } from '@angular/router';
import { HomeComponent } from './home/home.component';
import { VideoPlayerComponent } from './video-player/video-player.component';
import { AnalyticsComponent } from './pages/analytics/analytics';

export const routes: Routes = [
  { path: '',           component: HomeComponent },
  { path: 'watch/:id',  component: VideoPlayerComponent },
  { path: 'analytics',  component: AnalyticsComponent },
];