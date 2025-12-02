import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

@Component({
  selector: 'app-simulator',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './simulator.component.html',
  styleUrl: './simulator.component.scss',
})
export class SimulatorComponent {
  latitud: number | null = null;
  longitud: number | null = null;
  altura: number | null = null;
  modelo = '';
  formato = '';

  onGenerate() {
    // Por ahora solo mostramos en consola
    console.log('Generar distribución', {
      latitud: this.latitud,
      longitud: this.longitud,
      altura: this.altura,
      modelo: this.modelo,
    });
  }

  onDownload() {
    console.log('Descargar resultados en', this.formato);
  }
}
