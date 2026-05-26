import { Injectable } from '@angular/core';
import { BehaviorSubject, Observable, finalize } from 'rxjs';

import {
  BFieldResponse,
  CityResolveResponse,
  SimResponse,
  SimulatorApiService,
} from './simulator-api.service';
import { LanguageService } from '../i18n/language.service';

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

  private readonly initialState: SimulatorState;
  private readonly stateSubject: BehaviorSubject<SimulatorState>;

  readonly state$: Observable<SimulatorState>;

  constructor(
    private readonly simulatorApi: SimulatorApiService,
    private readonly i18n: LanguageService
  ) {
    this.initialState = this.createInitialState();
    this.stateSubject = new BehaviorSubject<SimulatorState>(this.loadInitialState());
    this.state$ = this.stateSubject.asObservable();
  }

  get snapshot(): SimulatorState {
    return this.stateSubject.value;
  }

  updateFormState(partial: Partial<EditableSimulatorFields>): void {
    this.patchState(partial);
  }

  resetState(): void {
    const initialState = this.createInitialState();
    this.stateSubject.next(initialState);
    this.persistState(initialState);
  }

  resolveCity(): void {
    const { city, country, altura, isResolvingCity } = this.snapshot;

    if (city.trim().length < 2 || isResolvingCity) {
      return;
    }

    this.patchState({
      isResolvingCity: true,
      locationStatus: this.i18n.t('sim.status.searchingCity'),
    });

    const payload = {
      city: city.trim(),
      country: country.trim(),
    };

    this.simulatorApi
      .resolveCity(payload)
      .pipe(
        finalize(() => {
          this.patchState({ isResolvingCity: false });
        })
      )
      .subscribe({
        next: (res: CityResolveResponse) => {
          this.patchState({
            lat: res.lat,
            lon: res.lon,
            resolvedDisplayName: res.display_name,
          });

          if (altura !== null) {
            this.patchState({
              locationStatus: this.i18n.t('sim.status.cityFoundUpdating'),
            });
            this.computeField(true);
            return;
          }

          this.patchState({
            locationStatus: this.i18n.t('sim.status.cityFoundNeedHeight'),
          });
        },
        error: () => {
          this.patchState({
            resolvedDisplayName: '',
            locationStatus: this.i18n.t('sim.status.cityNotFound'),
          });
        },
      });
  }

  computeField(silent = false): void {
    const { lat, lon, altura, isComputingField } = this.snapshot;

    if (lat === null || lon === null || altura === null || isComputingField) {
      return;
    }

    this.patchState({
      isComputingField: true,
      ...(silent ? {} : { locationStatus: this.i18n.t('sim.status.computingField') }),
    });

    const payload = { lat, lon, altura };

    this.simulatorApi
      .computeBField(payload)
      .pipe(
        finalize(() => {
          this.patchState({ isComputingField: false });
        })
      )
      .subscribe({
        next: (res: BFieldResponse) => {
          this.patchState({
            bx: res.bx,
            bz: res.bz,
            locationStatus: this.i18n.t('sim.status.fieldUpdated'),
          });
        },
        error: (err) => {
          this.patchState({
            locationStatus: err?.error?.detail ?? this.i18n.t('sim.status.fieldError'),
          });
        },
      });
  }

  startSimulation(): void {
    const { bx, bz, altura, isSimulating } = this.snapshot;

    if (bx === null || bz === null || altura === null || isSimulating) {
      return;
    }

    const payload = {
      bx,
      bz,
      altura,
      lang: this.i18n.currentLanguage(),
    };

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
      mensajeResultado: this.i18n.t('sim.status.simulationStarting'),
    });

    this.simulatorApi
      .simulateFull(payload)
      .pipe(
        finalize(() => {
          this.patchState({ isSimulating: false });
        })
      )
      .subscribe({
        next: (res: SimResponse) => {
          this.patchState({
            simulationDone: true,
            mensajeResultado: this.i18n.t('sim.status.simulationDone'),
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
          this.patchState({
            simulationDone: false,
            mensajeResultado:
              err?.error?.detail ?? this.i18n.t('sim.status.simulationError'),
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

  private createInitialState(): SimulatorState {
    return {
      city: '',
      country: '',

      lat: null,
      lon: null,
      altura: null,

      bx: null,
      bz: null,

      formato: '',

      mensajeResultado: this.i18n.t('sim.status.initialResult'),
      locationStatus: this.i18n.t('sim.status.initialLocation'),
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
    } catch {
      return this.initialState;
    }
  }

  private persistState(state: SimulatorState): void {
    try {
      sessionStorage.setItem(
        this.STORAGE_KEY,
        JSON.stringify(this.toPersistedState(state))
      );
    } catch {
      // proximamente jejeje
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
