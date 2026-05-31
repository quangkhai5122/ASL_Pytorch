import { useState, useEffect, useRef, useCallback } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import {
  Search, Star, StarOff, Filter, Hand, ChevronLeft, Loader2,
  Play, Pause, RotateCcw, Repeat, Gauge,
} from "lucide-react";
import { SkeletonCanvas, type RichFrameData } from "./SkeletonCanvas";

type RealDictionaryEntry = {
  id: string;
  gloss: string;
  hasVideo: boolean;
  hasSkeleton: boolean;
  // Fallbacks for mock UI:
  tags: string[];
  handedness: string;
  frequency: string;
  videoQuality: string;
  synonyms: string[];
  notes: string;
  examples: string[];
};

const SPEED_OPTIONS = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 2];

export function DictionarySearch() {
  const [query, setQuery] = useState("");
  const [allEntries, setAllEntries] = useState<RealDictionaryEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedEntry, setSelectedEntry] = useState<RealDictionaryEntry | null>(null);
  const [skeletonFrames, setSkeletonFrames] = useState<RichFrameData[] | null>(null);
  const [loadingSkeleton, setLoadingSkeleton] = useState(false);
  const [favorites, setFavorites] = useState<Set<string>>(new Set());
  const [filterOpen, setFilterOpen] = useState(false);
  const [filters, setFilters] = useState({ hasSkeleton: false });

  // ── Shared playback state ──
  const [isPlaying, setIsPlaying] = useState(true);
  const [isLooping, setIsLooping] = useState(true);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);
  const [skeletonFrame, setSkeletonFrame] = useState(0);
  const [speedMenuOpen, setSpeedMenuOpen] = useState(false);

  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    fetch("http://localhost:8000/api/v1/dictionary/")
      .then(r => r.json())
      .then(data => {
        const entries = data.words.map((w: any) => ({
          id: w.word,
          gloss: w.word,
          hasVideo: w.has_video,
          hasSkeleton: w.has_skeleton,
          tags: ["ASL"], 
          handedness: "all",
          frequency: "common",
          videoQuality: "high",
          synonyms: [],
          notes: "Dynamic data loaded from dataset.",
          examples: []
        }));
        setAllEntries(entries);
        setLoading(false);
      })
      .catch(e => {
        console.error("Failed to load dictionary", e);
        setLoading(false);
      });
  }, []);

  const handleSelectEntry = (entry: RealDictionaryEntry) => {
    setSelectedEntry(entry);
    setSkeletonFrames(null);
    setSkeletonFrame(0);
    setIsPlaying(true);
    setPlaybackSpeed(1);
    if (entry.hasSkeleton) {
      setLoadingSkeleton(true);
      fetch(`http://localhost:8000/api/v1/dictionary/${encodeURIComponent(entry.gloss)}/skeleton`)
        .then(r => r.json())
        .then(data => {
          setSkeletonFrames(data.frames);
          setLoadingSkeleton(false);
        })
        .catch(e => {
          console.error("Failed to load skeleton", e);
          setLoadingSkeleton(false);
        });
    }
  };

  const filteredResults = allEntries.filter(entry => {
    const matchesQuery = !query || entry.gloss.toLowerCase().includes(query.toLowerCase());
    const matchesSkeleton = !filters.hasSkeleton || entry.hasSkeleton;
    return matchesQuery && matchesSkeleton;
  });

  const toggleFav = (id: string) => {
    setFavorites(prev => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  };

  // ── Playback controls ──
  const handlePlayPause = useCallback(() => {
    const vid = videoRef.current;
    if (vid) {
      if (isPlaying) {
        vid.pause();
      } else {
        vid.play().catch(() => {});
      }
    }
    setIsPlaying(p => !p);
  }, [isPlaying]);

  const handleRestart = useCallback(() => {
    const vid = videoRef.current;
    if (vid) {
      vid.currentTime = 0;
      vid.play().catch(() => {});
    }
    setSkeletonFrame(0);
    setIsPlaying(true);
  }, []);

  const handleToggleLoop = useCallback(() => {
    setIsLooping(prev => {
      const next = !prev;
      if (videoRef.current) {
        videoRef.current.loop = next;
      }
      return next;
    });
  }, []);

  const handleSpeedChange = useCallback((speed: number) => {
    setPlaybackSpeed(speed);
    if (videoRef.current) {
      videoRef.current.playbackRate = speed;
    }
    setSpeedMenuOpen(false);
  }, []);

  const handleSkeletonFrameUpdate = useCallback((frame: number) => {
    if (!isLooping && skeletonFrames && frame === 0 && skeletonFrame > 0) {
      // Reached the end, stop
      setIsPlaying(false);
      return;
    }
    setSkeletonFrame(frame);
  }, [isLooping, skeletonFrames, skeletonFrame]);

  // Sync video events → playback state
  const handleVideoEnded = useCallback(() => {
    if (!isLooping) {
      setIsPlaying(false);
    }
  }, [isLooping]);

  // When video loads, sync loop + speed
  const handleVideoLoaded = useCallback(() => {
    const vid = videoRef.current;
    if (vid) {
      vid.loop = isLooping;
      vid.playbackRate = playbackSpeed;
      if (isPlaying) vid.play().catch(() => {});
    }
  }, [isLooping, playbackSpeed, isPlaying]);

  const totalSkeletonFrames = skeletonFrames?.length || 0;

  if (selectedEntry) {
    return (
      <div className="space-y-4 animate-fade-in">
        <Button 
          variant="ghost" 
          className="touch-target" 
          onClick={() => setSelectedEntry(null)} 
          aria-label="Back to search results"
        >
          <ChevronLeft className="w-4 h-4 mr-1" />Back to Dictionary
        </Button>

        <div className="asl-panel">
          <div className="asl-panel-header">
            <h2 className="text-2xl font-bold font-mono tracking-tight">{selectedEntry.gloss.toUpperCase()}</h2>
            <div className="flex items-center gap-2">
              <Button 
                size="icon" 
                variant="ghost" 
                onClick={() => toggleFav(selectedEntry.id)} 
                aria-label={favorites.has(selectedEntry.id) ? "Remove from favorites" : "Add to favorites"}
              >
                {favorites.has(selectedEntry.id) ? (
                  <Star className="w-6 h-6 text-warning fill-warning" />
                ) : (
                  <StarOff className="w-6 h-6 text-muted-foreground" />
                )}
              </Button>
            </div>
          </div>

          <div className="asl-panel-body space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Video Player */}
              <div className="space-y-2">
                <p className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">Reference Video</p>
                <div className="aspect-video bg-black rounded-xl overflow-hidden flex items-center justify-center border border-border shadow-inner">
                  {selectedEntry.hasVideo ? (
                    <video 
                      ref={videoRef}
                      key={selectedEntry.gloss}
                      src={`http://localhost:8000/api/v1/videos/${encodeURIComponent(selectedEntry.gloss)}.mp4`}
                      className="w-full h-full object-contain"
                      muted
                      loop={isLooping}
                      onLoadedData={handleVideoLoaded}
                      onEnded={handleVideoEnded}
                    />
                  ) : (
                    <div className="text-sm text-muted-foreground">No video available</div>
                  )}
                </div>
              </div>

              {/* Skeleton animation */}
              <div className="space-y-2">
                <p className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">Skeleton Animation</p>
                <div className="aspect-video bg-[#0a0a12] rounded-xl overflow-hidden flex items-center justify-center border border-border shadow-inner relative">
                  {selectedEntry.hasSkeleton ? (
                    loadingSkeleton ? (
                      <div className="flex flex-col items-center gap-3">
                        <Loader2 className="w-8 h-8 animate-spin text-primary" />
                        <span className="text-sm text-muted-foreground">Loading skeleton data...</span>
                      </div>
                    ) : skeletonFrames ? (
                      <div className="w-full h-full">
                        <SkeletonCanvas
                          framesData={skeletonFrames}
                          fps={20}
                          isPlaying={isPlaying}
                          playbackSpeed={playbackSpeed}
                          currentFrame={skeletonFrame}
                          onFrameUpdate={handleSkeletonFrameUpdate}
                        />
                      </div>
                    ) : (
                      <div className="text-sm text-muted-foreground">Failed to load skeleton</div>
                    )
                  ) : (
                    <div className="text-sm text-muted-foreground">No skeleton data available</div>
                  )}
                </div>
              </div>
            </div>

            {/* ══════════════════════════════════════════════════════════════
                UNIFIED PLAYBACK CONTROL BAR
               ══════════════════════════════════════════════════════════════ */}
            <div className="flex items-center justify-center gap-1 p-3 bg-muted/40 rounded-xl border border-border/50">
              {/* Restart */}
              <Button
                size="icon"
                variant="ghost"
                className="h-9 w-9 rounded-lg hover:bg-primary/10"
                onClick={handleRestart}
                aria-label="Restart from beginning"
                title="Restart"
              >
                <RotateCcw className="w-4 h-4" />
              </Button>

              {/* Play / Pause */}
              <Button
                size="icon"
                variant="default"
                className="h-11 w-11 rounded-full shadow-md"
                onClick={handlePlayPause}
                aria-label={isPlaying ? "Pause" : "Play"}
                title={isPlaying ? "Pause" : "Play"}
              >
                {isPlaying ? (
                  <Pause className="w-5 h-5" />
                ) : (
                  <Play className="w-5 h-5 ml-0.5" />
                )}
              </Button>

              {/* Loop toggle */}
              <Button
                size="icon"
                variant={isLooping ? "secondary" : "ghost"}
                className={`h-9 w-9 rounded-lg ${isLooping ? "bg-primary/15 text-primary" : "hover:bg-primary/10"}`}
                onClick={handleToggleLoop}
                aria-label={isLooping ? "Disable loop" : "Enable loop"}
                title={isLooping ? "Loop: ON" : "Loop: OFF"}
              >
                <Repeat className="w-4 h-4" />
              </Button>

              {/* Divider */}
              <div className="w-px h-6 bg-border mx-2" />

              {/* Speed selector */}
              <div className="relative">
                <Button
                  variant="outline"
                  size="sm"
                  className="h-9 rounded-lg text-xs font-mono gap-1.5 px-3"
                  onClick={() => setSpeedMenuOpen(p => !p)}
                  aria-label="Change playback speed"
                  title="Playback speed"
                >
                  <Gauge className="w-3.5 h-3.5" />
                  {playbackSpeed}x
                </Button>
                {speedMenuOpen && (
                  <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 bg-popover border border-border rounded-xl shadow-xl p-1.5 min-w-[100px] z-50 animate-in fade-in zoom-in-95 duration-150">
                    {SPEED_OPTIONS.map(speed => (
                      <button
                        key={speed}
                        className={`w-full text-left px-3 py-1.5 text-sm font-mono rounded-lg transition-colors ${
                          speed === playbackSpeed
                            ? "bg-primary/10 text-primary font-bold"
                            : "hover:bg-muted"
                        }`}
                        onClick={() => handleSpeedChange(speed)}
                      >
                        {speed}x
                      </button>
                    ))}
                  </div>
                )}
              </div>

              {/* Divider */}
              <div className="w-px h-6 bg-border mx-2" />

              {/* Frame counter */}
              {totalSkeletonFrames > 0 && (
                <span className="text-xs font-mono text-muted-foreground tabular-nums">
                  Frame {skeletonFrame + 1}/{totalSkeletonFrames}
                </span>
              )}
            </div>

            {/* Information Cards */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-4 border-t border-border">
              <div className="p-3 rounded-lg bg-muted/30">
                <p className="text-[10px] text-muted-foreground uppercase font-bold mb-1">Status</p>
                <p className="text-sm font-medium">Ready for Preview</p>
              </div>
              <div className="p-3 rounded-lg bg-muted/30">
                <p className="text-[10px] text-muted-foreground uppercase font-bold mb-1">Source</p>
                <p className="text-sm font-medium">WLASL Dataset</p>
              </div>
              <div className="p-3 rounded-lg bg-muted/30">
                <p className="text-[10px] text-muted-foreground uppercase font-bold mb-1">Frames</p>
                <p className="text-sm font-medium">{skeletonFrames?.length || 0} Points Tracked</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4 animate-in fade-in slide-in-from-bottom-2 duration-300">
      {/* Search & Filter Header */}
      <div className="asl-panel">
        <div className="asl-panel-body">
          <div className="flex flex-col md:flex-row gap-3">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" aria-hidden="true" />
              <Input
                value={query}
                onChange={e => setQuery(e.target.value)}
                placeholder="Search ASL dictionary (e.g. 'hello', 'thank you')..."
                className="pl-10 h-12 text-base rounded-xl shadow-sm focus-visible:ring-primary"
                aria-label="Search dictionary"
              />
            </div>
            <Button 
              variant={filterOpen ? "secondary" : "outline"} 
              className="h-12 px-6 rounded-xl flex items-center gap-2" 
              onClick={() => setFilterOpen(!filterOpen)}
            >
              <Filter className="w-4 h-4" />
              <span>Filters</span>
            </Button>
          </div>
          
          {filterOpen && (
            <div className="mt-4 p-4 bg-muted/40 rounded-xl border border-border/50 animate-in zoom-in-95 duration-200">
              <div className="flex items-center gap-4">
                <label className="flex items-center gap-2 text-sm font-medium cursor-pointer group">
                  <input 
                    type="checkbox" 
                    checked={filters.hasSkeleton} 
                    onChange={e => setFilters(f => ({ ...f, hasSkeleton: e.target.checked }))} 
                    className="w-4 h-4 rounded border-input text-primary focus:ring-primary" 
                  />
                  <span className="group-hover:text-primary transition-colors">Only show signs with skeleton data</span>
                </label>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Results Grid */}
      {loading ? (
        <div className="py-20 flex flex-col items-center justify-center gap-4">
          <Loader2 className="w-10 h-10 animate-spin text-primary opacity-50" />
          <p className="text-muted-foreground font-medium">Loading dictionary...</p>
        </div>
      ) : (
        <>
          <div className="flex items-center justify-between px-1">
            <p className="text-xs font-bold text-muted-foreground uppercase tracking-widest">
              Found {filteredResults.length} signs
            </p>
          </div>
          
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3" role="list" aria-label="Dictionary search results">
            {filteredResults.map(entry => (
              <button
                key={entry.id}
                role="listitem"
                className="asl-panel group text-left hover:border-primary/50 hover:shadow-md hover:-translate-y-0.5 transition-all duration-200 cursor-pointer overflow-hidden"
                onClick={() => handleSelectEntry(entry)}
              >
                <div className="p-4 flex items-center gap-4">
                  <div className="w-12 h-12 rounded-xl bg-primary/5 group-hover:bg-primary/10 flex items-center justify-center transition-colors">
                    <Hand className="w-6 h-6 text-primary" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="font-mono font-bold text-base group-hover:text-primary transition-colors truncate">
                      {entry.gloss.toUpperCase()}
                    </p>
                    <div className="flex items-center gap-2 mt-1">
                      {entry.hasVideo && <span className="text-[10px] font-bold px-2 py-0.5 rounded bg-blue-500/10 text-blue-500 uppercase tracking-tighter">Video</span>}
                      {entry.hasSkeleton && <span className="text-[10px] font-bold px-2 py-0.5 rounded bg-green-500/10 text-green-500 uppercase tracking-tighter">Skeleton</span>}
                    </div>
                  </div>
                  <Button
                    size="icon"
                    variant="ghost"
                    className="h-8 w-8 rounded-full opacity-0 group-hover:opacity-100 transition-opacity"
                    onClick={e => { e.stopPropagation(); toggleFav(entry.id); }}
                  >
                    {favorites.has(entry.id) ? (
                      <Star className="w-4 h-4 text-warning fill-warning" />
                    ) : (
                      <StarOff className="w-4 h-4 text-muted-foreground" />
                    )}
                  </Button>
                </div>
              </button>
            ))}
          </div>

          {filteredResults.length === 0 && (
            <div className="py-20 text-center space-y-3 bg-muted/20 rounded-3xl border border-dashed border-border">
              <Search className="w-12 h-12 text-muted-foreground mx-auto opacity-20" />
              <p className="text-muted-foreground font-medium">No signs found matching "{query}"</p>
              <Button variant="link" onClick={() => setQuery("")}>Clear search</Button>
            </div>
          )}
        </>
      )}
    </div>
  );
}
