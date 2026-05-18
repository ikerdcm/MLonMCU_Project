#ifndef KWS_MFCC_TABLES_H
#define KWS_MFCC_TABLES_H

#define MFCC_FRAME_SIZE       480u
#define MFCC_FRAME_STEP       320u
#define MFCC_FFT_SIZE         512u
#define MFCC_FFT_BINS         257u
#define MFCC_MEL_BINS         40u
#define MFCC_COEFF_COUNT      10u
#define MFCC_FRAME_COUNT      49u
#define MFCC_AUDIO_SAMPLES    16000u
#define MFCC_OUTPUT_ELEMS     (49u * 10u)

#define MFCC_INPUT_SCALE      0.03684842586517334f
#define MFCC_INPUT_ZERO_POINT (-9)

// Small tables in OCRAM (read-only)
extern const float kMfccWindow[480];
extern const float kMfccGoertzelCoeff[257];
extern const float kMfccGoertzelCos[257];
extern const float kMfccGoertzelSin[257];
extern const float kMfccDctMatrix[400];
// Large mel matrix in SDRAM (41 KB)
extern const float kMfccMelMatrix[10280];

#endif /* KWS_MFCC_TABLES_H */