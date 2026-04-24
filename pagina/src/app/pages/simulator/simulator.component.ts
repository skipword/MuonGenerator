import { Component, OnDestroy, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Subscription } from 'rxjs';

import {
  DownloadFormat,
  SimulatorState,
  SimulatorStateService,
} from '../../services/simulator-state.service';

@Component({
  selector: 'app-simulator',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './simulator.component.html',
  styleUrl: './simulator.component.scss',
})
export class SimulatorComponent implements OnInit, OnDestroy {
  private stateSub?: Subscription;
  private state: SimulatorState;

  private generationStartMs: number | null = null;
  private waitingForFirstRenderedResult = false;

  totalDisplayTimeSec: number | null = null;

  constructor(private readonly simulatorState: SimulatorStateService) {
    this.state = this.simulatorState.snapshot;
  }

  ngOnInit(): void {
    this.stateSub = this.simulatorState.state$.subscribe((state) => {
      this.state = state;
    });
  }

  ngOnDestroy(): void {
    this.stateSub?.unsubscribe();
  }

  get city(): string {
    return this.state.city;
  }

  set city(value: string) {
    this.simulatorState.updateFormState({ city: value });
  }

  get country(): string {
    return this.state.country;
  }

  set country(value: string) {
    this.simulatorState.updateFormState({ country: value });
  }

  get lat(): number | null {
    return this.state.lat;
  }

  set lat(value: number | null) {
    this.simulatorState.updateFormState({ lat: value });
  }

  get lon(): number | null {
    return this.state.lon;
  }

  set lon(value: number | null) {
    this.simulatorState.updateFormState({ lon: value });
  }

  get altura(): number | null {
    return this.state.altura;
  }

  set altura(value: number | null) {
    this.simulatorState.updateFormState({ altura: value });
  }

  get bx(): number | null {
    return this.state.bx;
  }

  set bx(value: number | null) {
    this.simulatorState.updateFormState({ bx: value });
  }

  get bz(): number | null {
    return this.state.bz;
  }

  set bz(value: number | null) {
    this.simulatorState.updateFormState({ bz: value });
  }

  get formato(): DownloadFormat {
    return this.state.formato;
  }

  set formato(value: DownloadFormat) {
    this.simulatorState.updateFormState({ formato: value });
  }

  get mensajeResultado(): string {
    return this.state.mensajeResultado;
  }

  get locationStatus(): string {
    return this.state.locationStatus;
  }

  get resolvedDisplayName(): string {
    return this.state.resolvedDisplayName;
  }

  get isResolvingCity(): boolean {
    return this.state.isResolvingCity;
  }

  get isComputingField(): boolean {
    return this.state.isComputingField;
  }

  get isSimulating(): boolean {
    return this.state.isSimulating;
  }

  get simulationDone(): boolean {
    return this.state.simulationDone;
  }

  get resultImageUrls(): string[] {
    return this.state.resultImageUrls;
  }

  get resultImageLabels(): string[] {
    return this.state.resultImageLabels;
  }

  get currentImageIndex(): number {
    return this.state.currentImageIndex;
  }

  get currentImageUrl(): string {
    return this.resultImageUrls[this.currentImageIndex] ?? '';
  }

  get currentImageLabel(): string {
    return this.resultImageLabels[this.currentImageIndex] ?? '';
  }

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

  get formattedTotalDisplayTime(): string {
    if (this.totalDisplayTimeSec === null) {
      return '';
    }

    return this.totalDisplayTimeSec.toFixed(2).replace('.', ',');
  }

  onResolveCity(): void {
    this.simulatorState.resolveCity();
  }

  onComputeField(silent = false): void {
    this.simulatorState.computeField(silent);
  }

  maybeRecomputeField(): void {
    if (this.lat !== null && this.lon !== null && this.altura !== null) {
      this.simulatorState.computeField(true);
    }
  }

  onGenerate(): void {
    if (!this.isFormValid || this.isSimulating) {
      return;
    }

    this.generationStartMs = performance.now();
    this.waitingForFirstRenderedResult = true;
    this.totalDisplayTimeSec = null;

    this.simulatorState.startSimulation();
  }

  onResultImageLoad(): void {
    if (!this.waitingForFirstRenderedResult || this.generationStartMs === null) {
      return;
    }

    this.totalDisplayTimeSec =
      (performance.now() - this.generationStartMs) / 1000;

    this.waitingForFirstRenderedResult = false;
  }

  prevImage(): void {
    this.simulatorState.prevImage();
  }

  nextImage(): void {
    this.simulatorState.nextImage();
  }

  onDownload(): void {
    if (!this.simulationDone || !this.formato) {
      return;
    }

    const url = this.simulatorState.getDownloadUrl(this.formato);

    if (!url) {
      console.error('No hay URL de descarga para:', this.formato);
      return;
    }

    window.open(url, '_blank', 'noopener');
  }
}