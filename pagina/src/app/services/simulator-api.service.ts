import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { environment } from '../../environments/environment';

export type CityResolveRequest = {
  city: string;
  country: string;
};

export type CityResolveResponse = {
  query: string;
  display_name: string;
  lat: number;
  lon: number;
};

export type BFieldRequest = {
  lat: number;
  lon: number;
  altura: number;
};

export type BFieldResponse = {
  lat: number;
  lon: number;
  altura: number;
  bx: number;
  bz: number;
  computed_at_utc: string;
  model: string;
};

export type SimFullRequest = {
  bx: number;
  bz: number;
  altura: number;
  lang?: 'es' | 'en' | 'pt' | 'fr';
};

export type SimulationStartResponse = {
  message: string;
  job_id: string;
  task_arn: string | null;
  status: 'queued' | 'running';
  queued: boolean;
};

export type SimulationStatusResponse = {
  job_id: string;
  status: 'queued' | 'running' | 'completed' | 'failed' | 'unknown';
  message?: string;
  updated_at?: string;
  meta?: {
    simulation_time_s?: number;
    [key: string]: unknown;
  };
  image_urls?: string[];
  image_labels?: string[];
  download_urls?: {
    csv?: string;
    shw?: string;
    shw_zip?: string;
  };
};

export type DownloadLinkResponse = {
  job_id: string;
  artifact: string;
  url: string;
};

@Injectable({
  providedIn: 'root',
})
export class SimulatorApiService {
  private readonly apiBaseUrl = environment.apiBaseUrl.replace(/\/+$/, '');

  constructor(private readonly http: HttpClient) {}

  resolveCity(
    payload: CityResolveRequest
  ): Observable<CityResolveResponse> {
    return this.http.post<CityResolveResponse>(
      `${this.apiBaseUrl}/resolve-city`,
      payload
    );
  }

  computeBField(payload: BFieldRequest): Observable<BFieldResponse> {
    return this.http.post<BFieldResponse>(
      `${this.apiBaseUrl}/compute-bfield`,
      payload
    );
  }

  simulateAws(payload: SimFullRequest): Observable<SimulationStartResponse> {
    return this.http.post<SimulationStartResponse>(
      `${this.apiBaseUrl}/simulate-aws`,
      payload
    );
  }

  getSimulationStatus(jobId: string): Observable<SimulationStatusResponse> {
    return this.http.get<SimulationStatusResponse>(
      `${this.apiBaseUrl}/simulation/${jobId}`
    );
  }

  getDownloadLink(
    jobId: string,
    artifact: 'csv' | 'shw'
  ): Observable<DownloadLinkResponse> {
    return this.http.get<DownloadLinkResponse>(
      `${this.apiBaseUrl}/simulation/${jobId}/download?artifact=${artifact}`
    );
  }
}