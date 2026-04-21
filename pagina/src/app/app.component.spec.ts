import { TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';

import { AppComponent } from './app.component';

describe('AppComponent', () => {
  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [AppComponent],
      providers: [provideRouter([])],
    }).compileComponents();
  });

  it('should create the app', () => {
    const fixture = TestBed.createComponent(AppComponent);
    const app = fixture.componentInstance;

    expect(app).toBeTruthy();
  });

  it('should start with the menu closed', () => {
    const fixture = TestBed.createComponent(AppComponent);
    const app = fixture.componentInstance;

    expect(app.menuOpen).toBeFalse();
  });

  it('should toggle the menu state', () => {
    const fixture = TestBed.createComponent(AppComponent);
    const app = fixture.componentInstance;

    app.toggleMenu();
    expect(app.menuOpen).toBeTrue();

    app.toggleMenu();
    expect(app.menuOpen).toBeFalse();
  });

  it('should close the menu', () => {
    const fixture = TestBed.createComponent(AppComponent);
    const app = fixture.componentInstance;

    app.menuOpen = true;
    app.closeMenu();

    expect(app.menuOpen).toBeFalse();
  });

  it('should render the navbar brand', () => {
    const fixture = TestBed.createComponent(AppComponent);
    fixture.detectChanges();

    const compiled = fixture.nativeElement as HTMLElement;

    expect(compiled.querySelector('.navbar__brand')?.textContent).toContain(
      'Generador global del flujo de muones'
    );
  });
});