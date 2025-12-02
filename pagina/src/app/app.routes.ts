import { Routes } from '@angular/router';
import { HomeComponent } from './pages/home/home.component';
import { SimulatorComponent } from './pages/simulator/simulator.component';
import { DescriptionComponent } from './pages/description/description.component';
import { DownloadsComponent } from './pages/downloads/downloads.component';

export const routes: Routes = [
  { path: '', component: HomeComponent, pathMatch: 'full' },
  { path: 'simulador', component: SimulatorComponent },
  { path: 'descripcion', component: DescriptionComponent },
  { path: 'descargas', component: DownloadsComponent },
  { path: '**', redirectTo: '' },
];
