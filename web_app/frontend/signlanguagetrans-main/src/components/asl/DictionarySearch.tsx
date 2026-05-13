import { useState } from "react";
import { MOCK_DICTIONARY, type DictionaryEntry } from "@/lib/mockData";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Search, Star, StarOff, Filter, Hand, Play, ChevronLeft } from "lucide-react";
import { SkeletonPreview } from "./SkeletonPreview";

export function DictionarySearch() {
  const [query, setQuery] = useState("");
  const [selectedEntry, setSelectedEntry] = useState<DictionaryEntry | null>(null);
  const [favorites, setFavorites] = useState<Set<string>>(new Set());
  const [filterOpen, setFilterOpen] = useState(false);
  const [filters, setFilters] = useState({ handedness: "all", hasSkeleton: false });

  const filteredResults = MOCK_DICTIONARY.filter(entry => {
    const matchesQuery = !query || entry.gloss.toLowerCase().includes(query.toLowerCase()) ||
      entry.tags.some(t => t.toLowerCase().includes(query.toLowerCase()));
    const matchesHandedness = filters.handedness === "all" || entry.handedness === filters.handedness;
    const matchesSkeleton = !filters.hasSkeleton || entry.hasSkeleton;
    return matchesQuery && matchesHandedness && matchesSkeleton;
  });

  const toggleFav = (id: string) => {
    setFavorites(prev => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  };

  if (selectedEntry) {
    return (
      <div className="space-y-4 animate-fade-in">
        <Button variant="ghost" className="touch-target" onClick={() => setSelectedEntry(null)} aria-label="Back to search results">
          <ChevronLeft className="w-4 h-4 mr-1" />Back
        </Button>
        <div className="asl-panel">
          <div className="asl-panel-header">
            <h2 className="text-lg font-bold font-mono">{selectedEntry.gloss}</h2>
            <Button size="icon" variant="ghost" onClick={() => toggleFav(selectedEntry.id)} aria-label={favorites.has(selectedEntry.id) ? "Remove from favorites" : "Add to favorites"}>
              {favorites.has(selectedEntry.id) ? <Star className="w-5 h-5 text-warning fill-warning" /> : <StarOff className="w-5 h-5 text-muted-foreground" />}
            </Button>
          </div>
          <div className="asl-panel-body space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Video placeholder */}
              <div>
                <p className="text-xs text-muted-foreground mb-2">Video</p>
                <div className="aspect-video bg-foreground/5 rounded-lg flex items-center justify-center relative">
                  <Play className="w-10 h-10 text-muted-foreground/30" aria-hidden="true" />
                  <span className="absolute bottom-2 left-2 text-[10px] text-muted-foreground">Sample video placeholder</span>
                </div>
              </div>
              {/* Skeleton animation */}
              <div>
                <p className="text-xs text-muted-foreground mb-2">Skeleton Animation</p>
                <div className="aspect-video bg-foreground/5 rounded-lg overflow-hidden">
                  {selectedEntry.hasSkeleton ? (
                    <SkeletonPreview animated />
                  ) : (
                    <div className="flex items-center justify-center h-full text-xs text-muted-foreground">No skeleton data</div>
                  )}
                </div>
              </div>
            </div>
            {/* Metadata */}
            <div className="grid grid-cols-2 gap-3 text-sm">
              <div><span className="text-muted-foreground">Handedness:</span> <span className="font-medium capitalize">{selectedEntry.handedness}</span></div>
              <div><span className="text-muted-foreground">Frequency:</span> <span className="font-medium capitalize">{selectedEntry.frequency}</span></div>
              <div><span className="text-muted-foreground">Tags:</span> <span className="font-medium">{selectedEntry.tags.join(", ")}</span></div>
              <div><span className="text-muted-foreground">Quality:</span> <span className="font-medium capitalize">{selectedEntry.videoQuality}</span></div>
            </div>
            {selectedEntry.synonyms.length > 0 && (
              <div className="text-sm"><span className="text-muted-foreground">Synonyms:</span> <span className="font-mono font-medium">{selectedEntry.synonyms.join(", ")}</span></div>
            )}
            <div className="text-sm"><span className="text-muted-foreground">Notes:</span> {selectedEntry.notes}</div>
            {selectedEntry.examples.length > 0 && (
              <div className="text-sm">
                <span className="text-muted-foreground">Examples:</span>
                <ul className="list-disc list-inside mt-1">
                  {selectedEntry.examples.map((ex, i) => <li key={i} className="font-mono text-xs">{ex}</li>)}
                </ul>
              </div>
            )}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {/* Search */}
      <div className="asl-panel">
        <div className="asl-panel-body">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" aria-hidden="true" />
            <Input
              value={query}
              onChange={e => setQuery(e.target.value)}
              placeholder="Search signs…"
              className="pl-10 touch-target"
              aria-label="Search dictionary"
            />
          </div>
          <div className="mt-2 flex items-center gap-2">
            <Button variant="outline" size="sm" className="touch-target" onClick={() => setFilterOpen(!filterOpen)} aria-label="Toggle filters">
              <Filter className="w-3.5 h-3.5 mr-1" />Filters
            </Button>
          </div>
          {filterOpen && (
            <div className="mt-3 p-3 bg-muted/50 rounded-lg space-y-2 animate-fade-in">
              <div className="flex items-center gap-2 flex-wrap">
                <span className="text-xs text-muted-foreground">Handedness:</span>
                {["all", "one-handed", "two-handed"].map(h => (
                  <Button key={h} size="sm" variant={filters.handedness === h ? "default" : "outline"} className="text-xs touch-target" onClick={() => setFilters(f => ({ ...f, handedness: h }))}>
                    {h === "all" ? "All" : h}
                  </Button>
                ))}
              </div>
              <label className="flex items-center gap-2 text-xs cursor-pointer">
                <input type="checkbox" checked={filters.hasSkeleton} onChange={e => setFilters(f => ({ ...f, hasSkeleton: e.target.checked }))} className="rounded" />
                Has skeleton data only
              </label>
            </div>
          )}
        </div>
      </div>

      {/* Results */}
      <div className="space-y-2" role="list" aria-label="Dictionary search results">
        {filteredResults.map(entry => (
          <button
            key={entry.id}
            role="listitem"
            className="asl-panel w-full text-left hover:border-primary/30 transition-colors cursor-pointer"
            onClick={() => setSelectedEntry(entry)}
            aria-label={`View ${entry.gloss} details`}
          >
            <div className="p-3 flex items-center gap-3">
              <div className="w-10 h-10 rounded-md bg-primary/10 flex items-center justify-center flex-shrink-0">
                <Hand className="w-5 h-5 text-primary" aria-hidden="true" />
              </div>
              <div className="flex-1 min-w-0">
                <p className="font-mono font-semibold text-sm">{entry.gloss}</p>
                <div className="flex items-center gap-2 mt-0.5">
                  <span className="text-[10px] text-muted-foreground capitalize">{entry.handedness}</span>
                  {entry.tags.map(tag => (
                    <span key={tag} className="text-[10px] px-1.5 py-0.5 rounded-full bg-muted text-muted-foreground">{tag}</span>
                  ))}
                </div>
              </div>
              <Button
                size="icon"
                variant="ghost"
                className="h-8 w-8 flex-shrink-0"
                onClick={e => { e.stopPropagation(); toggleFav(entry.id); }}
                aria-label={favorites.has(entry.id) ? "Remove from favorites" : "Add to favorites"}
              >
                {favorites.has(entry.id) ? <Star className="w-4 h-4 text-warning fill-warning" /> : <StarOff className="w-4 h-4 text-muted-foreground" />}
              </Button>
            </div>
          </button>
        ))}
        {filteredResults.length === 0 && (
          <p className="text-sm text-muted-foreground text-center py-8">No results found</p>
        )}
      </div>
    </div>
  );
}
