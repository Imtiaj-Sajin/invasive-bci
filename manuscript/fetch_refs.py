"""Fetch reference metadata from OpenAlex (by DOI when known, else by title search) and write refs_meta.json."""
import json, time, urllib.parse, requests
REFS = [
 ("hochberg2006", "doi:10.1038/nature04970"),
 ("hochberg2012", "doi:10.1038/nature11076"),
 ("collinger2013", "doi:10.1016/S0140-6736(12)61816-9"),
 ("pandarinath2017", "doi:10.7554/eLife.18554"),
 ("willett2021", "doi:10.1038/s41586-021-03506-2"),
 ("willett2023", "doi:10.1038/s41586-023-06377-x"),
 ("card2024", "doi:10.1056/NEJMoa2314132"),
 ("wairagkar2025", "doi:10.1038/s41586-025-09127-3"),
 ("card2026", "doi:10.1038/s41591-026-04414-6"),
 ("hahn2026", "doi:10.1038/s41591-026-04530-3"),
 ("hahn2025pre", "doi:10.1101/2025.07.02.25330310"),
 ("simeral2011", "title:Neural control of cursor trajectory and click by a human with tetraplegia 1000 days after implant of an intracortical microelectrode array"),
 ("perge2013", "title:Intra-day signal instabilities affect decoding performance in an intracortical neural interface system"),
 ("jarosiewicz2015", "doi:10.1126/scitranslmed.aac7328"),
 ("sussillo2016", "doi:10.1038/ncomms13749"),
 ("degenhart2020", "doi:10.1038/s41551-020-0542-9"),
 ("gallego2020", "doi:10.1038/s41593-019-0555-4"),
 ("ma2023", "doi:10.7554/eLife.84296"),
 ("karpowicz2025", "doi:10.1038/s41467-025-59652-y"),
 ("wilson2025", "doi:10.1038/s41551-025-01536-z"),
 ("pun2024", "doi:10.1038/s42003-024-06784-4"),
 ("karpowicz2024falcon", "doi:10.1101/2024.09.15.613126"),
 ("sponheim2021", "doi:10.1088/1741-2552/ac3eaf"),
 ("barrese2013", "title:Failure mode analysis of silicon-based intracortical microelectrode arrays in non-human primates"),
 ("barrese2016", "title:Scanning electron microscopy of chronically implanted intracortical microelectrode arrays in non-human primates"),
 ("chestek2011", "title:Long-term stability of neural prosthetic control signals from silicon cortical arrays in rhesus macaque motor cortex"),
 ("forrest2025", "doi:10.1088/1741-2552/ae1bda"),
 ("patel2023", "title:Utah array characterization and histological analysis of a multi-year implant in non-human primate motor and sensory cortices"),
 ("bjanes2025", "title:Quantifying physical degradation alongside recording and stimulation performance of 980 intracortical microelectrodes chronically implanted in three humans for 956-2246 days"),
 ("chen2023", "title:Chronic stability of a neuroprosthesis comprising multiple adjacent Utah arrays in monkeys"),
 ("woeppel2021", "doi:10.3389/fbioe.2021.759711"),
 ("downey2018", "title:Intracortical recording stability in human brain-computer interface users"),
 ("bishop2014", "title:Self-recalibrating classifiers for intracortical brain-computer interfaces"),
 ("flint2013", "title:Long term, stable brain machine interface performance using local field potentials and multiunit spikes"),
 ("nason2020", "doi:10.1038/s41551-020-0591-0"),
 ("nason2021", "doi:10.1016/j.neuron.2021.08.009"),
 ("temmar2025", "title:LINK: Long-term Intracortical Neural activity and Kinematics"),
 ("perich2018", "doi:10.1016/j.neuron.2018.09.030"),
 ("gilja2012", "doi:10.1038/nn.3265"),
 ("orsborn2012", "title:Closed-loop decoder adaptation on intermediate time-scales facilitates rapid BMI performance improvements independent of decoder initialization conditions"),
 ("dangi2013", "title:Design and analysis of closed-loop decoder adaptation algorithms for brain-machine interfaces"),
 ("kuzborskij2013", "title:Stability and hypothesis transfer learning"),
 ("sugiyama2007", "title:Covariate shift adaptation by importance weighted cross validation"),
 ("patil2024", "title:Optimal ridge regularization for out-of-distribution prediction"),
 ("wan2023", "doi:10.3389/fncom.2023.1135783"),
 ("kozai2015", "title:Brain tissue responses to neural implants impact signal sensitivity and intervention strategies"),
 ("polikov2005", "title:Response of brain tissue to chronically implanted neural electrodes"),
 ("trautmann2019", "title:Accurate estimation of neural population dynamics without spike sorting"),
 ("willsey2022", "title:Real-time brain-machine interface in non-human primates achieves high-velocity prosthetic finger movements using a shallow feedforward neural network decoder"),
 ("pandarinath2018lfads", "doi:10.1038/s41592-018-0109-9"),
 ("ye2025ndt3", "doi:10.1101/2025.02.02.634313"),
 ("mensh2017", "doi:10.1371/journal.pcbi.1005619"),
]
out = {}
H = {"User-Agent": "mailto:imtiajsajin@gmail.com"}
for key, q in REFS:
    try:
        if q.startswith("doi:"):
            r = requests.get("https://api.openalex.org/works/https://doi.org/" + q[4:], headers=H, timeout=30).json()
        else:
            r = requests.get("https://api.openalex.org/works", params={"search": q[6:], "per-page": 1}, headers=H, timeout=30).json()["results"][0]
        b = r.get("biblio") or {}
        out[key] = dict(title=r.get("title"), year=r.get("publication_year"), doi=(r.get("doi") or "").replace("https://doi.org/", ""),
                        venue=((r.get("primary_location") or {}).get("source") or {}).get("display_name"),
                        volume=b.get("volume"), issue=b.get("issue"), first_page=b.get("first_page"), last_page=b.get("last_page"),
                        authors=[a["author"]["display_name"] for a in r.get("authorships", [])], cited_by=r.get("cited_by_count"))
    except Exception as e:
        out[key] = {"error": str(e)}
    time.sleep(0.15)
json.dump(out, open("refs_meta.json", "w"), indent=1)
for k, v in out.items():
    if "error" in v: print("ERR", k, v["error"]); continue
    print(f"{k:20s} {v['year']} | {str(v['venue'])[:30]:30s} | {v['volume']} {v['first_page']} | {len(v['authors'])} au | {v['title'][:80]}")
