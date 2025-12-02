import { Component } from '@angular/core';
import { 
  RouterOutlet, 
  RouterLink,
  RouterLinkActive,
} from '@angular/router';

@Component({
  selector: 'app-root',
  imports: [RouterOutlet, RouterLink, RouterLinkActive],
  standalone: true,
  templateUrl: './app.component.html',
  styleUrl: './app.component.scss'
})
export class AppComponent {
  title = 'pagina';
}
