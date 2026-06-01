import { Injectable, signal } from '@angular/core';
import { Language, LanguageOption } from './i18n.types';
import { TRANSLATIONS } from './translations';

@Injectable({
  providedIn: 'root'
})
export class LanguageService {
  private readonly storageKey = 'selectedLanguage';

  readonly languages: LanguageOption[] = [
    { code: 'es', label: 'Español' },
    { code: 'en', label: 'English' },
    { code: 'pt', label: 'Português' },
    { code: 'fr', label: 'Français' }
  ];

  readonly currentLanguage = signal<Language>(this.getInitialLanguage());

  setLanguage(language: Language): void {
    this.currentLanguage.set(language);

    if (typeof localStorage !== 'undefined') {
      localStorage.setItem(this.storageKey, language);
    }

    this.setDocumentLanguage(language);
  }

  t(key: string): string {
    const lang = this.currentLanguage();
    return TRANSLATIONS[lang]?.[key] ?? TRANSLATIONS.es[key] ?? key;
  }

  private getInitialLanguage(): Language {
    if (typeof localStorage !== 'undefined') {
      const savedLanguage = localStorage.getItem(this.storageKey);

      if (savedLanguage && this.isSupportedLanguage(savedLanguage)) {
        this.setDocumentLanguage(savedLanguage);
        return savedLanguage;
      }
    }

    this.setDocumentLanguage('en');
    return 'en';
  }

  private setDocumentLanguage(language: Language): void {
    if (typeof document !== 'undefined') {
      document.documentElement.lang = language;
    }
  }

  private isSupportedLanguage(language: string): language is Language {
    return ['es', 'en', 'pt', 'fr'].includes(language);
  }
}
