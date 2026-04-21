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
};

export type SimResponse = {
  message: string;
  image_urls: string[];
  image_labels?: string[];
  download_csv_url?: string;
  download_shw_url?: string;
  run_id?: string;
  simulation_time_s?: number;
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

  simulateFull(payload: SimFullRequest): Observable<SimResponse> {
    return this.http.post<SimResponse>(
      `${this.apiBaseUrl}/simulate-full`,
      payload
    );
  }
}