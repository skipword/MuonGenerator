import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClient } from '@angular/common/http';
import { finalize } from 'rxjs/operators';

type SimResponse = {
  message: string;
  image_urls: string[];
  image_labels?: string[];
  download_csv_url?: string;
  download_shw_url?: string;
  run_id?: string;
  simulation_time_s?: number;
};

@Component({
  selector: 'app-simulator',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './simulator.component.html',
  styleUrl: './simulator.component.scss',
})
export class SimulatorComponent {
  bx: number | null = null;
  bz: number | null = null;
  altura: number | null = null;
  formato = '';

  mensajeResultado =
    'Complete el formulario y haga clic en "Generar distribución" para ver los resultados.';
  isSimulating = false;
  simulationDone = false;

  resultImageUrls: string[] = [];
  resultImageLabels: string[] = [];
  currentImageIndex = 0;

  private downloadUrls: Record<string, string> = {
    csv: '',
    shw: '',
  };

  private readonly API_BASE = 'http://localhost:8000';

  constructor(private http: HttpClient) {}

  get isFormValid(): boolean {
    return (
      this.bx !== null &&
      this.bz !== null &&
      this.altura !== null
    );
  }

  get currentImageUrl(): string {
    return this.resultImageUrls[this.currentImageIndex] ?? '';
  }

  get currentImageLabel(): string {
    return this.resultImageLabels[this.currentImageIndex] ?? '';
  }

  onGenerate() {
    if (!this.isFormValid || this.isSimulating) return;

    this.isSimulating = true;
    this.simulationDone = false;
    this.currentImageIndex = 0;
    this.resultImageUrls = [];
    this.resultImageLabels = [];
    this.downloadUrls = { csv: '', shw: '' };

    this.mensajeResultado = 'Ejecutando simulación… esto puede tardar unos segundos.';

    const payload = {
      bx: this.bx,
      bz: this.bz,
      altura: this.altura,
    };

    this.http
      .post<SimResponse>(`${this.API_BASE}/simulate-full`, payload)
      .pipe(finalize(() => (this.isSimulating = false)))
      .subscribe({
        next: (res) => {
          this.simulationDone = true;
          this.mensajeResultado = res.message;

          const ts = Date.now();
          this.resultImageUrls = (res.image_urls ?? []).map(
            (url) => `${url}?t=${ts}`
          );

          this.resultImageLabels = res.image_labels ?? [];

          this.downloadUrls['csv'] = res.download_csv_url ?? '';
          this.downloadUrls['shw'] = res.download_shw_url ?? '';
        },
        error: (err) => {
          this.simulationDone = false;
          this.mensajeResultado =
            'No se pudo completar la simulación (backend no disponible o error).';
          console.error(err);
        },
      });
  }

  prevImage() {
    if (this.currentImageIndex > 0) {
      this.currentImageIndex--;
    }
  }

  nextImage() {
    if (this.currentImageIndex < this.resultImageUrls.length - 1) {
      this.currentImageIndex++;
    }
  }

  onDownload() {
    if (!this.simulationDone || !this.formato) return;

    const url = this.downloadUrls[this.formato];
    if (!url) {
      console.error('No hay URL de descarga para:', this.formato);
      return;
    }

    const link = document.createElement('a');
    link.href = url;
    link.target = '_blank';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }
}