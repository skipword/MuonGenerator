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

type CityResolveResponse = {
  query: string;
  display_name: string;
  lat: number;
  lon: number;
};

type BFieldResponse = {
  lat: number;
  lon: number;
  altura: number;
  bx: number;
  bz: number;
  computed_at_utc: string;
  model: string;
};

@Component({
  selector: 'app-simulator',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './simulator.component.html',
  styleUrl: './simulator.component.scss',
})
export class SimulatorComponent {
  city = '';
  country = '';

  lat: number | null = null;
  lon: number | null = null;
  altura: number | null = null;

  bx: number | null = null;
  bz: number | null = null;

  formato = '';

  mensajeResultado =
    'Complete el formulario y haga clic en "Generar distribución" para ver los resultados.';

  locationStatus =
    'Busque una ciudad para autocompletar latitud y longitud. Luego ajuste la altura y actualice Bx/Bz.';
  resolvedDisplayName = '';

  isResolvingCity = false;
  isComputingField = false;
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
    return this.bx !== null && this.bz !== null && this.altura !== null;
  }

  get canResolveCity(): boolean {
    return this.city.trim().length >= 2 && !this.isResolvingCity;
  }

  get canComputeField(): boolean {
    return (
      this.lat !== null &&
      this.lon !== null &&
      this.altura !== null &&
      !this.isComputingField
    );
  }

  get currentImageUrl(): string {
    return this.resultImageUrls[this.currentImageIndex] ?? '';
  }

  get currentImageLabel(): string {
    return this.resultImageLabels[this.currentImageIndex] ?? '';
  }

  onResolveCity() {
    if (!this.canResolveCity) return;

    this.isResolvingCity = true;
    this.locationStatus = 'Buscando ciudad...';

    const payload = {
      city: this.city.trim(),
      country: this.country.trim(),
    };

    this.http
      .post<CityResolveResponse>(`${this.API_BASE}/resolve-city`, payload)
      .pipe(finalize(() => (this.isResolvingCity = false)))
      .subscribe({
        next: (res) => {
          this.lat = res.lat;
          this.lon = res.lon;
          this.resolvedDisplayName = res.display_name;

          if (this.altura !== null) {
            this.locationStatus =
              'Ciudad encontrada. Actualizando automáticamente Bx y Bz...';
            this.onComputeField(true);
          } else {
            this.locationStatus =
              'Ciudad encontrada. Ahora ingrese o ajuste la altura y luego actualice Bx/Bz.';
          }
        },
        error: (err) => {
          this.resolvedDisplayName = '';
          this.locationStatus = 'No se pudo encontrar la ciudad indicada.';
          console.error(err);
        },
      });
  }

  onComputeField(silent = false) {
    if (!this.canComputeField) return;

    this.isComputingField = true;
    if (!silent) {
      this.locationStatus = 'Calculando Bx y Bz...';
    }

    const payload = {
      lat: this.lat,
      lon: this.lon,
      altura: this.altura,
    };

    this.http
      .post<BFieldResponse>(`${this.API_BASE}/compute-bfield`, payload)
      .pipe(finalize(() => (this.isComputingField = false)))
      .subscribe({
        next: (res) => {
          this.bx = res.bx;
          this.bz = res.bz;
          this.locationStatus =
            'Bx y Bz actualizados. Puede ajustarlos manualmente si necesita un valor más específico.';
        },
        error: (err) => {
          const backendDetail = err?.error?.detail;
          this.locationStatus = backendDetail || 'No se pudo calcular Bx y Bz.';
          console.error(err);
        },
      });
  }

  maybeRecomputeField() {
    if (this.lat !== null && this.lon !== null && this.altura !== null) {
      this.onComputeField(true);
    }
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