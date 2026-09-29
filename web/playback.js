// Frames owed for the time elapsed, so playback speed holds on any display refresh rate.
export function framesDue(elapsedMs, stepMs) {
  return Math.max(0, Math.floor(elapsedMs / stepMs));
}
