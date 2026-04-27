export const curatedSources = {
  earlyStop: {
    finalBest: "result/early_stop_dynamics_router_conservative_submit.json",
    highlightedCandidates: [
      "result/early_stop_dynamics_v1.json",
      "result/early_stop_dynamics_v2_local.json",
      "result/early_stop_mean_confidence_plus_dyn_conservative_trimmed_mean_logprob.json"
    ],
    autoPatterns: [
      "result/early_stop_dynamics*.json",
      "result/early_stop_mean_confidence*.json"
    ]
  },
  bestOfN: {
    finalBest: "result/best_of_n_nad_mixed_v2_aime_top2_gap1e3_logprob_submit.json",
    highlightedCandidates: [
      "result/best_of_n_nad_mixed_v1_complete.json",
      "result/best_of_n_nad_mixed_v2_aime_top3_selfcert_submit.json",
      "result/best_of_n_nad_mixed_v2_aime_top2_gap1e3_selfcert_submit.json"
    ],
    autoPatterns: [
      "result/*best_of_n*",
      "result/*mixed_v2*",
      "scripts/*best_of_n*"
    ]
  }
} as const;
