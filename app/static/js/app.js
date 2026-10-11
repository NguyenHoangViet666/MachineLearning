/**
 * app.js - Client-side interaction logic for MovieLens Cosine Recommender.
 */

document.addEventListener("DOMContentLoaded", () => {
  // Tự động nhận diện port: nếu chạy trên port 8000 thì dùng relative path, nếu mở bằng Live Server (5500) thì gọi sang 8000
  const API_BASE = (window.location.protocol === "file:" || window.location.port === "5500") ? "http://127.0.0.1:8000" : "";

  // Elements
  const navButtons = document.querySelectorAll(".nav-btn");
  const tabPanes = document.querySelectorAll(".tab-pane");

  const searchInput = document.getElementById("movie-search-input");
  const autocompleteList = document.getElementById("autocomplete-list");
  const searchSpinner = document.getElementById("search-spinner");
  const genreFilter = document.getElementById("filter-genre");
  const kSelect = document.getElementById("select-k");
  const kError = document.getElementById("k-error");
  function readK() {
    const k = kSelect.valueAsNumber;
    const valid = kSelect.checkValidity() && Number.isInteger(k) && k >= 1 && k <= 50;
    kSelect.setAttribute("aria-invalid", String(!valid));
    kError.textContent = valid ? "" : "K phải là số nguyên từ 1 đến 50 phim; không được để trống.";
    kError.classList.toggle("hidden", valid);
    return valid ? k : null;
  }
  const yearMin = document.getElementById("year-min");
  const yearMax = document.getElementById("year-max");

  async function apiFetch(url, options) {
    if (url.includes("/api/movies?") || url.includes("/api/recommendations?")) {
      for (const input of [yearMin, yearMax]) {
        if (input && !input.checkValidity()) throw new Error("Năm phải là số nguyên từ 1870 đến 2100.");
      }
      if (yearMin?.value && yearMax?.value && Number(yearMin.value) > Number(yearMax.value)) {
        throw new Error("Năm bắt đầu phải nhỏ hơn hoặc bằng năm kết thúc.");
      }
      if (yearMin?.value) url += `&year_min=${encodeURIComponent(yearMin.value)}`;
      if (yearMax?.value) url += `&year_max=${encodeURIComponent(yearMax.value)}`;
    }
    const response = await window.fetch(url, options);
    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(typeof error.detail === "string" ? error.detail : `Đầu vào không hợp lệ hoặc dịch vụ chưa sẵn sàng (HTTP ${response.status}).`);
    }
    return response;
  }
  const btnRecommend = document.getElementById("btn-search-recommend");
  const btnResetFilters = document.getElementById("btn-reset-filters");

  // Advanced Filter Panel Elements
  const btnToggleFilter = document.getElementById("btn-toggle-filter");
  const advancedFilterPanel = document.getElementById("advanced-filter-panel");
  const filterActiveCountBadge = document.getElementById("filter-active-count-badge");
  const btnSelectAllGenres = document.getElementById("btn-select-all-genres");
  const btnClearGenres = document.getElementById("btn-clear-genres");
  const filterCounterNumber = document.getElementById("filter-counter-number");
  const genresGridScroller = document.getElementById("genres-grid-scroller");
  const filterModeRadios = document.querySelectorAll('input[name="filter-mode-choice"]');
  const advFooterSummary = document.getElementById("adv-footer-summary");
  const btnPanelSearch = document.getElementById("btn-panel-search");
  const btnPanelReset = document.getElementById("btn-panel-reset");
  const btnPanelClose = document.getElementById("btn-panel-close");
  const activeGenresBar = document.getElementById("active-genres-bar");
  const activeModeIndicator = document.getElementById("active-mode-indicator");
  const activeGenresChips = document.getElementById("active-genres-chips");
  const btnClearAllTags = document.getElementById("btn-clear-all-tags");

  const seedCard = document.getElementById("seed-movie-card");
  const seedTitle = document.getElementById("seed-title");
  const seedGenres = document.getElementById("seed-genres");
  const seedRatings = document.getElementById("seed-ratings");
  const seedAvgRating = document.getElementById("seed-avg-rating");
  const seedWarningBox = document.getElementById("seed-warning-box");
  const seedWarningMsg = document.getElementById("seed-warning-msg");

  const recsGrid = document.getElementById("recommendations-grid");
  const recsCountTitle = document.getElementById("results-count-title");
  const queryMetaInfo = document.getElementById("query-meta-info");

  const failureTableBody = document.getElementById("failure-table-body");
  const btnReloadFailures = document.getElementById("btn-reload-failures");

  let selectedMovieId = null;
  let searchDebounceTimeout = null;
  const detailDialog = document.getElementById("movie-detail-dialog");
  const detailTitle = document.getElementById("movie-detail-title");
  const detailStatus = document.getElementById("movie-detail-status");
  const detailFields = document.getElementById("movie-detail-fields");
  const detailWarning = document.getElementById("movie-detail-warning");
  const detailRecommend = document.getElementById("movie-detail-recommend");
  let detailMovie = null;
  let detailRequest = 0;

  async function openMovieDetails(movieId) {
    const requestId = ++detailRequest;
    detailMovie = null;
    detailTitle.textContent = "Chi tiết phim";
    detailStatus.textContent = "Đang tải thông tin phim...";
    detailFields.classList.add("hidden");
    detailWarning.classList.add("hidden");
    detailRecommend.disabled = true;
    if (!detailDialog.open) detailDialog.showModal();
    try {
      const response = await apiFetch(`${API_BASE}/api/movies/${movieId}`);
      const movie = await response.json();
      if (requestId !== detailRequest || !detailDialog.open) return;
      detailMovie = movie;
      detailTitle.textContent = movie.title;
      detailStatus.textContent = "Thông tin phim và thống kê đánh giá từ tập train.";
      document.getElementById("movie-detail-id").textContent = movie.movieId;
      document.getElementById("movie-detail-year").textContent = movie.year ?? "Chưa rõ năm";
      document.getElementById("movie-detail-genres").textContent = movie.genres.replaceAll("|", ", ");
      document.getElementById("movie-detail-count").textContent = `${movie.rating_count.toLocaleString()} lượt`;
      document.getElementById("movie-detail-rating").textContent = movie.rating_count ? `${movie.rating_mean.toFixed(2)} / 5.0 sao` : "Chưa có đánh giá train";
      detailFields.classList.remove("hidden");
      detailWarning.textContent = movie.warning || "";
      detailWarning.classList.toggle("hidden", !movie.warning);
      detailRecommend.disabled = !movie.can_recommend;
    } catch (error) {
      if (requestId === detailRequest && detailDialog.open) detailStatus.textContent = error.message;
    }
  }
  document.getElementById("movie-detail-close").addEventListener("click", () => detailDialog.close());
  detailDialog.addEventListener("close", () => { ++detailRequest; detailMovie = null; });
  detailDialog.addEventListener("click", (event) => {
    const bounds = detailDialog.getBoundingClientRect();
    if (event.target === detailDialog && (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom)) detailDialog.close();
  });
  detailRecommend.addEventListener("click", () => {
    if (!detailMovie) return;
    const movie = detailMovie;
    detailDialog.close();
    selectedMovieId = movie.movieId;
    searchInput.value = movie.title;
    executeRecommendation(movie.movieId);
  });
  seedTitle.addEventListener("click", () => { if (selectedMovieId) openMovieDetails(selectedMovieId); });
  document.addEventListener("click", (event) => {
    const button = event.target.closest("[data-movie-detail]");
    if (button) openMovieDetails(Number(button.dataset.movieDetail));
  });

  async function loadEvaluation() {
    const body = document.getElementById("comparison-table-body");
    try {
      const response = await apiFetch(`${API_BASE}/api/model-card`);
      const card = await response.json();
      const metrics = card.evaluation_summary.test;
      const k = card.technical_specifications.k;
      document.getElementById("selected-algorithm").textContent = `Mô hình chọn bằng validation: Cosine ${card.technical_specifications.mean_centered ? "Mean-centered" : "Raw"}; K=${k}; ngưỡng rating train=${card.technical_specifications.min_ratings}.`;
      document.getElementById("selected-cold-start").textContent = `Cold-start: phim có ít hơn ${card.technical_specifications.min_ratings} lượt đánh giá train chưa đủ dữ liệu để phục vụ.`;
      const selected = metrics.selected;
      document.getElementById("kpi-hit-label").textContent = `HitRate@${k} (mô hình đã chọn)`;
      document.getElementById("kpi-precision-label").textContent = `Precision@${k}`;
      document.getElementById("kpi-hitrate").textContent = `${(selected[`HitRate@${k}`] * 100).toFixed(2)}%`;
      document.getElementById("kpi-precision").textContent = selected[`Precision@${k}`].toFixed(4);
      document.getElementById("kpi-coverage").textContent = `${(selected.Coverage * 100).toFixed(2)}%`;
      document.getElementById("kpi-latency").textContent = `${selected.AvgLatencyMs.toFixed(2)} ms`;
      body.innerHTML = [["popularity", "Popularity", "Số rating train"], ["cosine_raw", "Cosine Raw", "Rating thô"], ["cosine_centered", "Cosine Mean-centered", "Trừ trung bình user"]].map(([key, name, representation]) => {
        const row = metrics[key];
        return `<tr><td>${name}</td><td>${representation}</td><td>${row["HitRate@10"].toFixed(4)}</td><td>${row["Precision@10"].toFixed(4)}</td><td>${row["Recall@10"].toFixed(4)}</td><td>${row.Coverage.toFixed(4)}</td><td>${key === card.evaluation_summary.selected_model ? "Được chọn bằng validation" : "Mô hình đối chứng"}</td></tr>`;
      }).join("");
    } catch (error) {
      body.innerHTML = `<tr><td colspan="7">${escapeHtml(error.message)}</td></tr>`;
    }
  }

  // State: Bộ lọc thể loại đa năng
  const selectedGenres = new Set();
  let genreMatchMode = "any"; // 'any' | 'exclude' | 'all'

  // ==================== TAB NAVIGATION ====================
  navButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      navButtons.forEach(b => b.classList.remove("active"));
      tabPanes.forEach(p => p.classList.remove("active"));

      btn.classList.add("active");
      const targetTabId = btn.getAttribute("data-tab");
      const targetPane = document.getElementById(targetTabId);
      if (targetPane) {
        targetPane.classList.add("active");
      }

      if (targetTabId === "tab-dashboard") {
        loadFailureCases();
        loadEvaluation();
      }
      if (targetTabId === "tab-scope") loadEvaluation();
    });
  });

  // ==================== AUTOCOMPLETE SEARCH ====================
  searchInput.addEventListener("input", (e) => {
    const val = e.target.value.trim();
    selectedMovieId = null;
    clearTimeout(searchDebounceTimeout);

    if (val.length < 2) {
      autocompleteList.classList.add("hidden");
      autocompleteList.innerHTML = "";
      return;
    }

    searchDebounceTimeout = setTimeout(async () => {
      searchSpinner.classList.remove("hidden");
      try {
        let url = `${API_BASE}/api/movies?query=${encodeURIComponent(val)}&limit=10`;
        if (selectedGenres.size > 0) {
          url += `&genres=${encodeURIComponent(Array.from(selectedGenres).join(","))}&match_mode=${genreMatchMode}`;
        }
        const res = await apiFetch(url);
        const data = await res.json();
        renderAutocomplete(data.movies || []);
      } catch (err) {
        console.error("Lỗi tìm kiếm phim:", err);
        autocompleteList.innerHTML = `<div class="autocomplete-item">${escapeHtml(err.message)}</div>`;
        autocompleteList.classList.remove("hidden");
      } finally {
        searchSpinner.classList.add("hidden");
      }
    }, 250);
  });

  function renderAutocomplete(movies) {
    if (!movies || movies.length === 0) {
      autocompleteList.innerHTML = `<div class="autocomplete-item"><span class="ac-title">Không tìm thấy phim phù hợp</span></div>`;
      autocompleteList.classList.remove("hidden");
      return;
    }

    autocompleteList.innerHTML = movies.map(m => `
      <div class="autocomplete-item" data-id="${m.movieId}" data-title="${escapeHtml(m.title)}">
        <span class="ac-title">${escapeHtml(m.title)}</span>
        <span class="ac-genres">${escapeHtml(m.genres)}</span>
      </div>
    `).join("");

    autocompleteList.classList.remove("hidden");

    // Click chọn phim từ autocomplete
    autocompleteList.querySelectorAll(".autocomplete-item").forEach(item => {
      item.addEventListener("click", () => {
        const id = parseInt(item.getAttribute("data-id"));
        const title = item.getAttribute("data-title");
        if (id) {
          selectedMovieId = id;
          searchInput.value = title;
          autocompleteList.classList.add("hidden");
          executeRecommendation(selectedMovieId);
          openMovieDetails(id);
        }
      });
    });
  }

  // Click ra ngoài thì đóng dropdown autocomplete
  document.addEventListener("click", (e) => {
    if (!searchInput.contains(e.target) && !autocompleteList.contains(e.target)) {
      autocompleteList.classList.add("hidden");
    }
  });

  // ==================== BỘ LỌC THỂ LOẠI NÂNG CAO ====================
  // Đảm bảo các cell thể loại được bind sự kiện
  function bindGenreCellEvents() {
    document.querySelectorAll(".genre-item-cell").forEach(cell => {
      cell.onclick = () => {
        const g = cell.getAttribute("data-genre");
        if (!g) return;
        if (selectedGenres.has(g)) {
          selectedGenres.delete(g);
        } else {
          selectedGenres.add(g);
        }
        updateFilterUI();
      };
    });
  }

  // Nạp danh sách thể loại nếu mở bằng file tĩnh/Live Server mà chưa có trong HTML
  async function ensureGenresLoaded() {
    const existingCells = document.querySelectorAll(".genre-item-cell");
    if (existingCells.length === 0 && genresGridScroller) {
      try {
        const res = await apiFetch(`${API_BASE}/api/genres`);
        if (res.ok) {
          const data = await res.json();
          genresGridScroller.innerHTML = (data.genres || []).map(g => `
            <div class="genre-item-cell" data-genre="${escapeHtml(g.genre)}">
              <span class="genre-checkbox-box">
                <svg class="check-svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3"><polyline points="20 6 9 17 4 12"></polyline></svg>
              </span>
              <span class="genre-item-label">${escapeHtml(g.genre)}</span>
              <span class="genre-item-badge">${g.count}</span>
            </div>
          `).join("");
        }
      } catch (err) {
        console.warn("Không thể tải danh sách thể loại từ API:", err);
      }
    }
    bindGenreCellEvents();
  }

  // Mở/đóng panel Lọc Nâng Cao
  function toggleFilterPanel(forceOpen = null) {
    if (!advancedFilterPanel || !btnToggleFilter) return;
    const shouldOpen = forceOpen !== null ? forceOpen : advancedFilterPanel.classList.contains("hidden");
    if (shouldOpen) {
      advancedFilterPanel.classList.remove("hidden");
      btnToggleFilter.classList.add("open");
      btnToggleFilter.setAttribute("aria-expanded", "true");
    } else {
      advancedFilterPanel.classList.add("hidden");
      btnToggleFilter.classList.remove("open");
      btnToggleFilter.setAttribute("aria-expanded", "false");
    }
  }

  if (btnToggleFilter) {
    btnToggleFilter.addEventListener("click", (e) => {
      e.stopPropagation();
      toggleFilterPanel();
    });
  }

  if (btnPanelClose) {
    btnPanelClose.addEventListener("click", (e) => {
      e.stopPropagation();
      toggleFilterPanel(false);
    });
  }

  // Đóng panel khi click ra bên ngoài
  document.addEventListener("click", (e) => {
    if (advancedFilterPanel && !advancedFilterPanel.classList.contains("hidden")) {
      if (!advancedFilterPanel.contains(e.target) && !btnToggleFilter.contains(e.target)) {
        toggleFilterPanel(false);
      }
    }
  });

  // Nút Chọn tất cả thể loại
  if (btnSelectAllGenres) {
    btnSelectAllGenres.addEventListener("click", () => {
      document.querySelectorAll(".genre-item-cell").forEach(cell => {
        const g = cell.getAttribute("data-genre");
        if (g) selectedGenres.add(g);
      });
      updateFilterUI();
    });
  }

  // Nút Bỏ chọn tất cả thể loại
  if (btnClearGenres) {
    btnClearGenres.addEventListener("click", () => {
      selectedGenres.clear();
      updateFilterUI();
    });
  }

  // Nút Xóa toàn bộ bộ lọc trên thanh active-genres-bar
  if (btnClearAllTags) {
    btnClearAllTags.addEventListener("click", () => {
      selectedGenres.clear();
      updateFilterUI();
      if (selectedMovieId) {
        executeRecommendation(selectedMovieId);
      }
    });
  }

  // Lắng nghe thay đổi 3 chế độ (any, exclude, all)
  filterModeRadios.forEach(radio => {
    radio.addEventListener("change", () => {
      if (radio.checked) {
        genreMatchMode = radio.value;
        document.querySelectorAll(".filter-mode-item").forEach(item => {
          item.classList.remove("active");
        });
        const parentLabel = radio.closest(".filter-mode-item");
        if (parentLabel) parentLabel.classList.add("active");
        updateFilterUI();
      }
    });
  });

  // Cập nhật giao diện bộ lọc: số đếm, active tags, thông báo tóm tắt
  function updateFilterUI() {
    const totalSelected = selectedGenres.size;
    const totalGenres = document.querySelectorAll(".genre-item-cell").length || 19;

    // 1. Cập nhật số đếm trên header panel
    if (filterCounterNumber) {
      filterCounterNumber.textContent = totalSelected;
    }
    const countTotalEl = document.querySelector(".adv-filter-counter .count-total");
    if (countTotalEl) {
      countTotalEl.textContent = `/ ${totalGenres} đã chọn`;
    }

    // 2. Cập nhật badge trên nút "Lọc nâng cao"
    if (filterActiveCountBadge) {
      if (totalSelected > 0) {
        filterActiveCountBadge.textContent = totalSelected;
        filterActiveCountBadge.classList.remove("hidden");
      } else {
        filterActiveCountBadge.classList.add("hidden");
      }
    }

    // 3. Đồng bộ trạng thái .checked trên từng cell thể loại
    document.querySelectorAll(".genre-item-cell").forEach(cell => {
      const g = cell.getAttribute("data-genre");
      if (selectedGenres.has(g)) {
        cell.classList.add("checked");
      } else {
        cell.classList.remove("checked");
      }
    });

    // 4. Cập nhật dòng tóm tắt ở footer của dropdown
    if (advFooterSummary) {
      if (totalSelected === 0) {
        advFooterSummary.textContent = "Chưa chọn thể loại nào (tìm kiếm toàn bộ kho phim)";
      } else if (genreMatchMode === "any") {
        advFooterSummary.textContent = `Đã chọn ${totalSelected} thể loại: Tìm phim có ít nhất 1 thể loại này (OR)`;
      } else if (genreMatchMode === "exclude") {
        advFooterSummary.textContent = `Đã chọn ${totalSelected} thể loại: Loại trừ các phim có thể loại này (NOT)`;
      } else if (genreMatchMode === "all") {
        advFooterSummary.textContent = `Đã chọn ${totalSelected} thể loại: Phim bắt buộc có đủ tất cả các thể loại này (AND)`;
      }
    }

    // 5. Cập nhật thanh Active Tags hiển thị bên dưới thanh tìm kiếm
    if (activeGenresBar) {
      if (totalSelected > 0) {
        activeGenresBar.classList.remove("hidden");
        if (activeModeIndicator) {
          if (genreMatchMode === "any") {
            activeModeIndicator.textContent = "Logic: Có 1 trong các thể loại (OR)";
          } else if (genreMatchMode === "exclude") {
            activeModeIndicator.textContent = "Logic: Loại trừ thể loại đã chọn (NOT)";
          } else if (genreMatchMode === "all") {
            activeModeIndicator.textContent = "Logic: Phải có đủ các thể loại (AND)";
          }
        }

        if (activeGenresChips) {
          activeGenresChips.innerHTML = Array.from(selectedGenres).map(g => `
            <span class="active-genre-tag">
              <span>${escapeHtml(g)}</span>
              <button type="button" class="tag-remove-btn" data-genre="${escapeHtml(g)}" title="Bỏ chọn ${escapeHtml(g)}">&times;</button>
            </span>
          `).join("");

          activeGenresChips.querySelectorAll(".tag-remove-btn").forEach(btn => {
            btn.onclick = (e) => {
              e.stopPropagation();
              const genreToRemove = btn.getAttribute("data-genre");
              if (genreToRemove) {
                selectedGenres.delete(genreToRemove);
                updateFilterUI();
                if (selectedMovieId) {
                  executeRecommendation(selectedMovieId);
                }
              }
            };
          });
        }
      } else {
        activeGenresBar.classList.add("hidden");
        if (activeGenresChips) activeGenresChips.innerHTML = "";
      }
    }

    // Đồng bộ với select hidden filter-genre (tương thích ngược)
    if (genreFilter) {
      if (totalSelected === 1 && genreMatchMode === "any") {
        genreFilter.value = Array.from(selectedGenres)[0];
      } else {
        genreFilter.value = "all";
      }
    }
  }

  // ==================== TÌM KIẾM & LÀM MỚI ====================
  async function handleSearchAction() {
    if (readK() === null) return;
    if (selectedMovieId) {
      await executeRecommendation(selectedMovieId);
      return;
    }
    try {
      const query = searchInput.value.trim();
      if (!query && !selectedGenres.size && !yearMin?.value && !yearMax?.value) {
        alert("Nhập tên phim, chọn bộ lọc hoặc chọn một phim demo để bắt đầu.");
        return;
      }
      const url = `${API_BASE}/api/movies?query=${encodeURIComponent(query)}&genres=${encodeURIComponent(Array.from(selectedGenres).join(","))}&match_mode=${genreMatchMode}&limit=24`;
      const response = await apiFetch(url);
      const data = await response.json();
      renderAutocomplete(data.movies);
    } catch (error) {
      alert(error.message);
    }
  }
  // Đặt lại toàn bộ bộ lọc và từ khóa tìm kiếm
  function resetAllFilters() {
    searchInput.value = "";
    if (yearMin) yearMin.value = "";
    if (yearMax) yearMax.value = "";
    selectedGenres.clear();
    genreMatchMode = "any";

    // Đặt lại radio mode về 'any'
    const radioAny = document.querySelector('input[name="filter-mode-choice"][value="any"]');
    if (radioAny) {
      radioAny.checked = true;
      document.querySelectorAll(".filter-mode-item").forEach(item => item.classList.remove("active"));
      const labelAny = document.getElementById("label-mode-any");
      if (labelAny) labelAny.classList.add("active");
    }

    if (kSelect) kSelect.value = "10";
    updateFilterUI();

    // Reset về phim mẫu mặc định: Toy Story (movieId: 1)
    selectedMovieId = 1;
    executeRecommendation(1);
  }

  // Gắn sự kiện các nút Tìm Kiếm
  if (btnRecommend) {
    btnRecommend.addEventListener("click", handleSearchAction);
  }
  if (btnPanelSearch) {
    btnPanelSearch.addEventListener("click", () => {
      toggleFilterPanel(false);
      handleSearchAction();
    });
  }

  // Gắn sự kiện các nút Làm Mới
  if (btnResetFilters) {
    btnResetFilters.addEventListener("click", resetAllFilters);
  }
  if (btnPanelReset) {
    btnPanelReset.addEventListener("click", resetAllFilters);
  }

  // ==================== QUICK SEED CHIPS ====================
  document.querySelectorAll(".seed-chip").forEach(chip => {
    chip.addEventListener("click", () => {
      const id = parseInt(chip.getAttribute("data-id"));
      const title = chip.getAttribute("data-title");
      selectedMovieId = id;
      searchInput.value = title;
      autocompleteList.classList.add("hidden");
      executeRecommendation(id);
    });
  });

  if (kSelect) {
    kSelect.addEventListener("input", readK);
    kSelect.addEventListener("change", () => {
      if (selectedMovieId) executeRecommendation(selectedMovieId);
    });
  }

  // ==================== THỰC THI GỢI Ý (COSINE + FILTER) ====================
  async function executeRecommendation(movieId) {
    const k = readK();
    if (k === null) return;
    
    // Hiển thị trạng thái đang tải
    recsGrid.innerHTML = `
      <div class="empty-state">
        <div class="spinner" style="position:static; margin:0 auto 1rem; width:36px; height:36px;"></div>
        <h4>Đang tính toán độ tương đồng Cosine...</h4>
        <p>Đang truy vấn ma trận thưa Item-User trên không gian 610 chiều và áp dụng bộ lọc thể loại kết hợp.</p>
      </div>
    `;

    const startTime = performance.now();

    try {
      // 1. Lấy chi tiết seed movie
      const movieRes = await apiFetch(`${API_BASE}/api/movies/${movieId}`);
      if (movieRes.ok) {
        const movieData = await movieRes.json();
        seedTitle.textContent = movieData.title;
        seedGenres.textContent = movieData.genres;
        seedRatings.textContent = movieData.rating_count ? movieData.rating_count.toLocaleString() : "0";
        seedAvgRating.textContent = movieData.rating_mean ? movieData.rating_mean.toFixed(1) + " / 5.0" : "N/A";
        seedCard.classList.remove("hidden");
      }

      // 2. Lấy gợi ý có tích hợp lọc thể loại nâng cao
      let url = `${API_BASE}/api/recommendations?movie_id=${movieId}&k=${k}`;
      if (selectedGenres.size > 0) {
        const genresParam = Array.from(selectedGenres).join(",");
        url += `&genres=${encodeURIComponent(genresParam)}&match_mode=${genreMatchMode}`;
      } else if (genreFilter && genreFilter.value && genreFilter.value !== "all") {
        url += `&genre=${encodeURIComponent(genreFilter.value)}`;
      }

      const recRes = await apiFetch(url);
      const recData = await recRes.json();
      const elapsed = Math.round(performance.now() - startTime);

      let filterDesc = "Toàn bộ";
      if (selectedGenres.size > 0) {
        const modeLabel = genreMatchMode === "any" ? "OR" : genreMatchMode === "exclude" ? "NOT" : "AND";
        filterDesc = `${selectedGenres.size} thể loại (${modeLabel})`;
      }
      queryMetaInfo.textContent = `Độ trễ API: ${elapsed}ms | K=${k} | Bộ lọc: ${filterDesc}`;

      // Xử lý cảnh báo (Warning) nếu dữ liệu thưa
      if (recData.warning) {
        seedWarningMsg.textContent = recData.warning;
        seedWarningBox.classList.remove("hidden");
      } else {
        seedWarningBox.classList.add("hidden");
      }

      renderRecommendations(recData.recommendations || []);
    } catch (err) {
      console.error("Lỗi khi tải gợi ý:", err);
      recsGrid.innerHTML = `
        <div class="empty-state">
          <div class="empty-icon">⚠️</div>
          <h4>Lỗi xử lý yêu cầu</h4>
          <p>${escapeHtml(err.message || "Vui lòng đảm bảo server đang chạy và model đã được nạp.")}</p>
        </div>
      `;
    }
  }

  function renderRecommendations(items) {
    if (!items || items.length === 0) {
      const modeText = genreMatchMode === "any" ? "chứa ít nhất 1 thể loại" : genreMatchMode === "exclude" ? "loại trừ thể loại" : "chứa đầy đủ các thể loại";
      recsGrid.innerHTML = `
        <div class="empty-state">
          <div class="empty-icon">🔍</div>
          <h4>Không tìm thấy phim tương đồng phù hợp</h4>
          <p>Phim này có thể có quá ít tương tác hoặc không khớp với bộ lọc thể loại (${modeText}) đã chọn.</p>
        </div>
      `;
      recsCountTitle.textContent = "Không có kết quả";
      return;
    }

    recsCountTitle.textContent = `Tìm thấy ${items.length} phim tương đồng hàng đầu`;

    recsGrid.innerHTML = items.map((item, idx) => {
      const simPercent = Math.min(100, Math.max(0, Math.round(item.similarity_score * 100)));

      // Render genres thành các badge pill với highlight nếu khớp với thể loại đang chọn
      const genresList = (item.genres || "").split(/[|,]/).map(g => g.trim()).filter(Boolean);
      const genresHtml = genresList.map(g => {
        const isMatched = selectedGenres.has(g);
        return `<span class="genre-pill ${isMatched ? 'matched' : ''}">${escapeHtml(g)}</span>`;
      }).join("");

      return `
        <div class="rec-card">
          <div>
            <div class="rec-top">
              <span class="rec-rank">#${idx + 1}</span>
              <div class="rec-score-box">
                <span class="rec-score-val">${item.similarity_score.toFixed(4)}</span>
                <span class="rec-score-lbl">Cosine Sim</span>
              </div>
            </div>
            <h4 class="rec-title"><button type="button" class="movie-detail-link" data-movie-detail="${item.movieId}" title="Xem chi tiết phim">${escapeHtml(item.title)}</button></h4>
            <div class="rec-genres">${genresHtml}</div>
            
            <div class="sim-bar-container" title="Độ tương đồng: ${simPercent}%">
              <div class="sim-bar-fill" style="width: ${simPercent}%"></div>
            </div>

            <div class="rec-reason">
              💡 ${escapeHtml(item.reason)}
            </div>
          </div>

          <div class="rec-bottom">
            <span>⭐ Rating: <strong>${item.rating_mean.toFixed(1)}</strong>/5.0</span>
            <button type="button" class="btn-select-seed" data-id="${item.movieId}" data-title="${escapeHtml(item.title)}" title="Chọn làm phim hạt giống để tìm phim tương tự">
              <span>Đổi hạt giống</span> ↗
            </button>
          </div>
        </div>
      `;
    }).join("");

    // Gắn sự kiện "Đổi hạt giống" trên các card gợi ý
    recsGrid.querySelectorAll(".btn-select-seed").forEach(btn => {
      btn.addEventListener("click", () => {
        const id = parseInt(btn.getAttribute("data-id"));
        const title = btn.getAttribute("data-title");
        if (id) {
          selectedMovieId = id;
          searchInput.value = title;
          executeRecommendation(id);
          window.scrollTo({ top: seedCard.offsetTop - 80, behavior: "smooth" });
        }
      });
    });
  }

  // ==================== DASHBOARD: 10 FAILURE CASES ====================
  async function loadFailureCases() {
    try {
      const res = await apiFetch(`${API_BASE}/api/failure-cases`);
      if (!res.ok) return;
      const data = await res.json();
      
      failureTableBody.innerHTML = data.map(item => `
        <tr>
          <td><code>${item.movieId}</code></td>
          <td><strong>${escapeHtml(item.title)}</strong></td>
          <td><span class="badge ${item.case_type.includes('Bom') ? 'badge-primary' : 'badge-warning'}">${escapeHtml(item.case_type)}</span></td>
          <td>${escapeHtml(item.top_1_sim)}</td>
          <td><code>${typeof item.top_1_score === 'number' ? item.top_1_score.toFixed(4) : item.top_1_score}</code></td>
          <td><small style="color:var(--text-dim);">${escapeHtml(item.identified_issues.join(" | "))}</small></td>
        </tr>
      `).join("");
    } catch (err) {
      console.warn("Chưa tải được bảng 10 ca lỗi:", err);
    }
  }

  if (btnReloadFailures) {
    btnReloadFailures.addEventListener("click", loadFailureCases);
  }

  function escapeHtml(str) {
    if (!str) return "";
    return str.toString()
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  // ==================== CINEBOT IN-HOUSE NLP CLIENT LOGIC ====================
  const cinebotFab = document.getElementById("cinebot-fab");
  const cinebotWindow = document.getElementById("cinebot-window");
  const cinebotBtnClose = document.getElementById("cinebot-btn-close");
  const cinebotBtnClear = document.getElementById("cinebot-btn-clear");
  const cinebotForm = document.getElementById("cinebot-form");
  const cinebotInput = document.getElementById("cinebot-input");
  const cinebotMessages = document.getElementById("cinebot-messages");
  const cinebotTyping = document.getElementById("cinebot-typing");
  const cinebotQuickChips = document.getElementById("cinebot-quick-chips");
  const navBtnChat = document.getElementById("nav-btn-chat");

  const initialBotWelcome = `
    <div class="chat-msg bot-msg">
      <div class="msg-bubble">
        <p>👋 <strong>Chào bạn! Tôi là CineBot</strong> — Trợ lý ảo điện ảnh được phát triển hoàn toàn nội bộ bằng thuật toán <strong>TF-IDF & Cosine Similarity</strong>.</p>
        <p>Tôi có thể giúp bạn tìm phim tương tự, khám phá theo tâm trạng hoặc giải thích công thức toán học của hệ thống. Bạn muốn tìm phim gì hôm nay?</p>
      </div>
    </div>
  `;

  function toggleCinebot(forceOpen = null) {
    if (!cinebotWindow) return;
    const shouldOpen = forceOpen !== null ? forceOpen : !cinebotWindow.classList.contains("open");
    if (shouldOpen) {
      cinebotWindow.classList.add("open");
      setTimeout(() => { cinebotInput?.focus(); }, 150);
    } else {
      cinebotWindow.classList.remove("open");
    }
  }

  if (cinebotFab) {
    cinebotFab.addEventListener("click", () => toggleCinebot());
  }

  if (navBtnChat) {
    navBtnChat.addEventListener("click", () => toggleCinebot(true));
  }

  if (cinebotBtnClose) {
    cinebotBtnClose.addEventListener("click", () => toggleCinebot(false));
  }

  if (cinebotBtnClear) {
    cinebotBtnClear.addEventListener("click", () => {
      if (cinebotMessages) {
        cinebotMessages.innerHTML = initialBotWelcome;
      }
    });
  }

  // Quick Chips
  if (cinebotQuickChips) {
    cinebotQuickChips.addEventListener("click", (e) => {
      const chip = e.target.closest(".chip-btn");
      if (chip && chip.dataset.query) {
        sendChatQuery(chip.dataset.query);
      }
    });
  }

  function appendUserMessage(text) {
    const div = document.createElement("div");
    div.className = "chat-msg user-msg";
    div.innerHTML = `<div class="msg-bubble"><p>${escapeHtml(text)}</p></div>`;
    cinebotMessages.appendChild(div);
    scrollChatBottom();
  }

  function formatMarkdown(text) {
    if (!text) return "";
    let html = escapeHtml(text);
    // Bold: **text**
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    // Italic: *text*
    html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');
    // Math formulas: $formula$
    html = html.replace(/\$(.*?)\$/g, '<code style="color:var(--accent-secondary);">$1</code>');
    // Blockquote: &gt; text
    html = html.replace(/^&gt;\s*(.*)$/gm, '<blockquote>$1</blockquote>');
    // Unordered lists: - item
    html = html.replace(/^[•\-]\s*(.*)$/gm, '<li>$1</li>');
    html = html.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');
    // Newlines
    html = html.replace(/\n\n/g, '</p><p>');
    html = html.replace(/\n/g, '<br>');
    return `<p>${html}</p>`;
  }

  function appendBotMessage(replyHtml, recommendations = []) {
    const div = document.createElement("div");
    div.className = "chat-msg bot-msg";

    let recsHtml = "";
    if (recommendations && recommendations.length > 0) {
      recsHtml = `<div class="chat-movie-cards">` + recommendations.map(rec => `
        <div class="chat-movie-item">
          <div class="chat-movie-top">
            <button type="button" class="chat-movie-title movie-detail-link" data-movie-detail="${rec.movieId}">${escapeHtml(rec.title)}</button>
            <span class="chat-similarity-badge">${typeof rec.similarity_score === 'number' ? (rec.similarity_score * 100).toFixed(1) + '%' : rec.similarity_score}</span>
          </div>
          <div class="chat-movie-genres">${escapeHtml(rec.genres)}</div>
          <div class="chat-movie-actions">
            <button class="chat-action-btn btn-chat-ask-similar" data-title="${escapeHtml(rec.title)}">
              🔍 Tìm phim tương tự
            </button>
            <button class="chat-action-btn btn-chat-view-main" data-id="${rec.movieId}">
              ↗ Xem trên web
            </button>
          </div>
        </div>
      `).join("") + `</div>`;
    }

    div.innerHTML = `<div class="msg-bubble">${replyHtml}${recsHtml}</div>`;
    cinebotMessages.appendChild(div);

    // Gắn sự kiện cho các nút hành động trong thẻ phim
    div.querySelectorAll(".btn-chat-ask-similar").forEach(btn => {
      btn.addEventListener("click", () => {
        const title = btn.getAttribute("data-title");
        sendChatQuery(`Gợi ý phim giống ${title}`);
      });
    });

    div.querySelectorAll(".btn-chat-view-main").forEach(btn => {
      btn.addEventListener("click", () => {
        const mId = parseInt(btn.getAttribute("data-id"));
        if (mId) {
          // Chuyển sang Tab 1 và thực thi gợi ý
          const recommenderNavBtn = document.querySelector('[data-tab="tab-recommender"]');
          if (recommenderNavBtn) recommenderNavBtn.click();
          selectedMovieId = mId;
          executeRecommendation(mId);
          window.scrollTo({ top: 0, behavior: 'smooth' });
        }
      });
    });

    scrollChatBottom();
  }

  function scrollChatBottom() {
    if (cinebotMessages) {
      cinebotMessages.scrollTop = cinebotMessages.scrollHeight;
    }
  }

  async function sendChatQuery(text) {
    if (!text || !text.trim()) return;
    const cleanText = text.trim();

    appendUserMessage(cleanText);
    if (cinebotInput) cinebotInput.value = "";

    if (cinebotTyping) cinebotTyping.style.display = "flex";
    scrollChatBottom();

    try {
      const res = await apiFetch(`${API_BASE}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: cleanText, k: 5 })
      });

      if (!res.ok) {
        throw new Error(`Server error: ${res.status}`);
      }

      const data = await res.json();
      if (cinebotTyping) cinebotTyping.style.display = "none";

      const formattedReply = formatMarkdown(data.reply);
      appendBotMessage(formattedReply, data.recommendations);
    } catch (err) {
      console.error("CineBot Error:", err);
      if (cinebotTyping) cinebotTyping.style.display = "none";
      appendBotMessage("<p>⚠️ <em>Đã có sự cố kết nối với CineBot. Vui lòng kiểm tra lại server hoặc thử lại sau!</em></p>");
    }
  }

  if (cinebotForm) {
    cinebotForm.addEventListener("submit", (e) => {
      e.preventDefault();
      const val = cinebotInput?.value;
      if (val) {
        sendChatQuery(val);
      }
    });
  }

  // Khởi tạo bộ lọc thể loại & tự động load phim mẫu mặc định: Toy Story (movieId: 1)
  ensureGenresLoaded();
  updateFilterUI();
  selectedMovieId = 1;
  executeRecommendation(1);
});
