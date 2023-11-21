number_of_testcases = 1

SINR2BER_table = np.zeros((number_of_testcases,4))

for i in range(number_of_testcases):
    ch_SINR = int(random.uniform(ch_SINR_min, ch_SINR_max))
    tx_gain_i = int(random.uniform(tx_gain_min, tx_gain_max))
    RX_Samples = radio_channel(use_sdr=use_sdr, tx_signal = TX_Samples, tx_gain = tx_gain_i, rx_gain = rx_gain, ch_SINR = ch_SINR)
    symbol_index=OFDM_SDR_Functions_torch.sync_TTI(TX_Samples, RX_Samples, config.leading_zeros)
    SINR = OFDM_SDR_Functions_torch.SINR(RX_Samples, 100, symbol_index)
    RX_NO_CP = OFDM_SDR_Functions_torch.CP_removal(RX_Samples, symbol_index, config.S, config.FFT_size, config.CP, plotsig=False)
    RX_NO_CP = RX_NO_CP / torch.max(torch.abs(RX_NO_CP))
    OFDM_demod = OFDM_SDR_Functions_torch.DFT(RX_NO_CP, plotDFT=False)
    H_estim = OFDM_SDR_Functions_torch.channelEstimate_LS(TTI_mask_RE, pilot_symbols, config.F, config.FFT_offset, config.Sp, OFDM_demod, plotEst=False)
    OFDM_demod_no_offsets = OFDM_SDR_Functions_torch.remove_fft_Offests(OFDM_demod, config.F, config.FFT_offset)
    equalized_H_estim = OFDM_SDR_Functions_torch.equalize_ZF(OFDM_demod_no_offsets, H_estim, config.F, config.S)
    QAM_est = OFDM_SDR_Functions_torch.get_payload_symbols(TTI_mask_RE, equalized_H_estim, config.FFT_offset, config.F, plotQAM=False)
    PS_est, hardDecision = OFDM_SDR_Functions_torch.Demapping(QAM_est, de_mapping_table_Qm)
    bits_est = OFDM_SDR_Functions_torch.PS(PS_est)
    error_count = torch.sum(bits_est != pdsch_bits.flatten()).float()  # Count of unequal bits
    error_rate = error_count / bits_est.numel()  # Error rate calculation
    BER = torch.round(error_rate * 1000) / 1000  # Round to 3 decimal places
    SINR2BER_table[i,0] = round(tx_gain_i,1)
    SINR2BER_table[i,1] = round(rx_gain,1)
    SINR2BER_table[i,2] = round(SINR,1)
    SINR2BER_table[i,3] = BER
