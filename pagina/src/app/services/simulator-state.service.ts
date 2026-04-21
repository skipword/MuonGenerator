import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { BehaviorSubject, finalize } from 'rxjs';

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

type SimResponse = {
  message: string;
  image_urls: string[];
  image_labels?: string[];
  download_csv_url?: string;
  download_shw_url?: string;
  run_id?: string;
  simulation_time_s?: number;
};

export type DownloadFormat = '' | 'csv' | 'shw';

export type SimulatorState = {
  city: string;
  country: string;

  lat: number | null;
  lon: number | null;
  altura: number | null;

  bx: number | null;
  bz: number | null;

  formato: DownloadFormat;

  mensajeResultado: string;
  locationStatus: string;
  resolvedDisplayName: string;

  isResolvingCity: boolean;
  isComputingField: boolean;
  isSimulating: boolean;
  simulationDone: boolean;

  resultImageUrls: string[];
  resultImageLabels: string[];
  currentImageIndex: number;

  downloadUrls: {
    csv: string;
    shw: string;
  };

  runId: string;
};

type EditableSimulatorFields = Pick<
  SimulatorState,
  'city' | 'country' | 'lat' | 'lon' | 'altura' | 'bx' | 'bz' | 'formato'
>;

type PersistedSimulatorState = Pick<
  SimulatorState,
  | 'city'
  | 'country'
  | 'lat'
  | 'lon'
  | 'altura'
  | 'bx'
  | 'bz'
  | 'formato'
  | 'mensajeResultado'
  | 'locationStatus'
  | 'resolvedDisplayName'
  | 'simulationDone'
  | 'resultImageUrls'
  | 'resultImageLabels'
  | 'currentImageIndex'
  | 'downloadUrls'
  | 'runId'
>;

@Injectable({
  providedIn: 'root',
})
export class SimulatorStateService {
  private readonly STORAGE_KEY = 'simulator-state';

  // Idealmente esto debería venir de environment.ts,
  // pero lo dejo aquí para que puedas copiar y pegar solo estos 2 archivos.
  private readonly API_BASE = 'http://localhost:8000';

  private readonly initialState: SimulatorState = {
    city: '',
    country: '',

    lat: null,
    lon: null,
    altura: null,

    bx: null,
    bz: null,

    formato: '',

    mensajeResultado:
      'Complete el formulario y haga clic en "Generar distribución" para ver los resultados.',

    locationStatus:
      'Busque una ciudad para autocompletar latitud y longitud. Luego ajuste la altura y actualice Bx/Bz.',

    resolvedDisplayName: '',

    isResolvingCity: false,
    isComputingField: false,
    isSimulating: false,
    simulationDone: false,

    resultImageUrls: [],
    resultImageLabels: [],
    currentImageIndex: 0,

    downloadUrls: {
      csv: '',
      shw: '',
    },

    runId: '',
  };

  private readonly stateSubject = new BehaviorSubject<SimulatorState>(
    this.loadInitialState()
  );

  readonly state$ = this.stateSubject.asObservable();

  constructor(private readonly http: HttpClient) {}

  get snapshot(): SimulatorState {
    return this.stateSubject.value;
  }

  updateFormState(partial: Partial<EditableSimulatorFields>): void {
    this.patchState(partial);
  }

  resetState(): void {
    this.stateSubject.next(this.initialState);
    this.persistState(this.initialState);
  }

  resolveCity(): void {
    const { city, country, altura, isResolvingCity } = this.snapshot;

    if (city.trim().length < 2 || isResolvingCity) {
      return;
    }

    this.patchState({
      isResolvingCity: true,
      locationStatus: 'Buscando ciudad...',
    });

    const payload = {
      city: city.trim(),
      country: country.trim(),
    };

    this.http
      .post<CityResolveResponse>(`${this.API_BASE}/resolve-city`, payload)
      .pipe(
        finalize(() => {
          this.patchState({ isResolvingCity: false });
        })
      )
      .subscribe({
        next: (res) => {
          this.patchState({
            lat: res.lat,
            lon: res.lon,
            resolvedDisplayName: res.display_name,
          });

          if (altura !== null) {
            this.patchState({
              locationStatus:
                'Ciudad encontrada. Actualizando automáticamente Bx y Bz...',
            });
            this.computeField(true);
            return;
          }

          this.patchState({
            locationStatus:
              'Ciudad encontrada. Ahora ingrese o ajuste la altura y luego actualice Bx/Bz.',
          });
        },
        error: (err) => {
          console.error(err);

          this.patchState({
            resolvedDisplayName: '',
            locationStatus: 'No se pudo encontrar la ciudad indicada.',
          });
        },
      });
  }

  computeField(silent = false): void {
    const { lat, lon, altura, isComputingField } = this.snapshot;

    if (
      lat === null ||
      lon === null ||
      altura === null ||
      isComputingField
    ) {
      return;
    }

    this.patchState({
      isComputingField: true,
      ...(silent ? {} : { locationStatus: 'Calculando Bx y Bz...' }),
    });

    const payload = { lat, lon, altura };

    this.http
      .post<BFieldResponse>(`${this.API_BASE}/compute-bfield`, payload)
      .pipe(
        finalize(() => {
          this.patchState({ isComputingField: false });
        })
      )
      .subscribe({
        next: (res) => {
          this.patchState({
            bx: res.bx,
            bz: res.bz,
            locationStatus:
              'Bx y Bz actualizados. Puede ajustarlos manualmente si necesita un valor más específico.',
          });
        },
        error: (err) => {
          console.error(err);

          this.patchState({
            locationStatus:
              err?.error?.detail ?? 'No se pudo calcular Bx y Bz.',
          });
        },
      });
  }

  startSimulation(): void {
    const { bx, bz, altura, isSimulating } = this.snapshot;

    if (
      bx === null ||
      bz === null ||
      altura === null ||
      isSimulating
    ) {
      return;
    }

    const payload = { bx, bz, altura };

    this.patchState({
      isSimulating: true,
      simulationDone: false,
      currentImageIndex: 0,
      resultImageUrls: [],
      resultImageLabels: [],
      downloadUrls: {
        csv: '',
        shw: '',
      },
      runId: '',
      mensajeResultado:
        'Ejecutando simulación… esto puede tardar unos segundos.',
    });

    this.http
      .post<SimResponse>(`${this.API_BASE}/simulate-full`, payload)
      .pipe(
        finalize(() => {
          this.patchState({ isSimulating: false });
        })
      )
      .subscribe({
        next: (res) => {
          this.patchState({
            simulationDone: true,
            mensajeResultado: res.message,
            resultImageUrls: this.addCacheBusting(res.image_urls ?? []),
            resultImageLabels: res.image_labels ?? [],
            downloadUrls: {
              csv: res.download_csv_url ?? '',
              shw: res.download_shw_url ?? '',
            },
            runId: res.run_id ?? '',
            currentImageIndex: 0,
          });
        },
        error: (err) => {
          console.error(err);

          this.patchState({
            simulationDone: false,
            mensajeResultado:
              err?.error?.detail ??
              'No se pudo completar la simulación (backend no disponible o error).',
          });
        },
      });
  }

  prevImage(): void {
    const { currentImageIndex } = this.snapshot;

    if (currentImageIndex <= 0) {
      return;
    }

    this.patchState({
      currentImageIndex: currentImageIndex - 1,
    });
  }

  nextImage(): void {
    const { currentImageIndex, resultImageUrls } = this.snapshot;

    if (currentImageIndex >= resultImageUrls.length - 1) {
      return;
    }

    this.patchState({
      currentImageIndex: currentImageIndex + 1,
    });
  }

  getDownloadUrl(format: DownloadFormat): string {
    if (!format) {
      return '';
    }

    return this.snapshot.downloadUrls[format];
  }

  private patchState(partial: Partial<SimulatorState>): void {
    const nextState: SimulatorState = {
      ...this.snapshot,
      ...partial,
      downloadUrls: {
        ...this.snapshot.downloadUrls,
        ...(partial.downloadUrls ?? {}),
      },
    };

    this.stateSubject.next(nextState);
    this.persistState(nextState);
  }

  private loadInitialState(): SimulatorState {
    try {
      const raw = sessionStorage.getItem(this.STORAGE_KEY);

      if (!raw) {
        return this.initialState;
      }

      const parsed = JSON.parse(raw) as Partial<PersistedSimulatorState>;

      return {
        ...this.initialState,
        ...parsed,
        isResolvingCity: false,
        isComputingField: false,
        isSimulating: false,
        downloadUrls: {
          ...this.initialState.downloadUrls,
          ...(parsed.downloadUrls ?? {}),
        },
      };
    } catch (error) {
      console.error('No se pudo restaurar el estado del simulador:', error);
      return this.initialState;
    }
  }

  private persistState(state: SimulatorState): void {
    try {
      sessionStorage.setItem(
        this.STORAGE_KEY,
        JSON.stringify(this.toPersistedState(state))
      );
    } catch (error) {
      console.error('No se pudo guardar el estado del simulador:', error);
    }
  }

  private toPersistedState(state: SimulatorState): PersistedSimulatorState {
    return {
      city: state.city,
      country: state.country,
      lat: state.lat,
      lon: state.lon,
      altura: state.altura,
      bx: state.bx,
      bz: state.bz,
      formato: state.formato,
      mensajeResultado: state.mensajeResultado,
      locationStatus: state.locationStatus,
      resolvedDisplayName: state.resolvedDisplayName,
      simulationDone: state.simulationDone,
      resultImageUrls: state.resultImageUrls,
      resultImageLabels: state.resultImageLabels,
      currentImageIndex: state.currentImageIndex,
      downloadUrls: state.downloadUrls,
      runId: state.runId,
    };
  }

  private addCacheBusting(urls: string[]): string[] {
    const timestamp = Date.now();

    return urls.map((url) =>
      url.includes('?') ? `${url}&t=${timestamp}` : `${url}?t=${timestamp}`
    );
  }
}