import { Component, inject } from '@angular/core';
import { LanguageService } from '../../i18n/language.service';

@Component({
  selector: 'app-downloads',
  imports: [],
  templateUrl: './downloads.component.html',
  styleUrl: './downloads.component.scss'
})
export class DownloadsComponent {
  readonly i18n = inject(LanguageService);
}