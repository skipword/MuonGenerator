import { Component, inject } from '@angular/core';
import {
  RouterLink,
  RouterLinkActive,
  RouterOutlet
} from '@angular/router';
import { FormsModule } from '@angular/forms';
import { LanguageService } from './i18n/language.service';
import { Language } from './i18n/i18n.types';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet, RouterLink, RouterLinkActive, FormsModule],
  templateUrl: './app.component.html',
  styleUrl: './app.component.scss',
})
export class AppComponent {
  readonly i18n = inject(LanguageService);

  menuOpen = false;

  get selectedLanguage(): Language {
    return this.i18n.currentLanguage();
  }

  set selectedLanguage(language: Language) {
    this.i18n.setLanguage(language);
  }

  toggleMenu(): void {
    this.menuOpen = !this.menuOpen;
  }

  closeMenu(): void {
    this.menuOpen = false;
  }
}