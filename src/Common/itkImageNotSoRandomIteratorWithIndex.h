#ifndef __itkImageNotSoRandomIteratorWithIndex_h
#define __itkImageNotSoRandomIteratorWithIndex_h

/** Make sure vnl/vnl_sample.h is not included by 
 * itkImageRandomIteratorWithIndex.txx */
#ifndef vnl_sample_h_
	#define vnl_sample_h_
	#define __dont_forget_to_undef_vnl_sample_h_
#endif

/** The functions that replace the functions in vnl_sample.h */
#include "elx_sample.h"

#define vnl_sample_reseed elx_sample_reseed
#define vnl_sample_uniform elx_sample_uniform
#define ImageRandomIteratorWithIndex ImageNotSoRandomIteratorWithIndex

/** Make sure itkImageRandomIteratorWithIndex.h is really included here */
#if defined(__itkImageRandomIteratorWithIndex_h)
	#undef __itkImageRandomIteratorWithIndex_h
#else 
	#define __dont_forget_to_undef_itkit_h_
#endif	

#if defined(_itkImageRandomIteratorWithIndex_txx)
	#undef _itkImageRandomIteratorWithIndex_txx
#else 
	#define __dont_forget_to_undef_itkit_txx_
#endif	

/** include the itk-version. */
#include "itkImageRandomIteratorWithIndex.h"

/** clean up. */

/* Make sure the normal itkRandomIterator still can be used */
#ifdef __dont_forget_to_undef_itkit_h_
	#undef __dont_forget_to_undef_itkit_h_
	#undef __itkImageRandomIteratorWithIndex_h
#endif 

#ifdef __dont_forget_to_undef_itkit_txx_
	#undef __dont_forget_to_undef_itkit_txx_
	#undef _itkImageRandomIteratorWithIndex_txx
#endif
 
#undef vnl_sample_reseed
#undef vnl_sample_uniform
#undef ImageRandomIteratorWithIndex

#ifdef __dont_forget_to_undef_vnl_sample_h_
#undef __dont_forget_to_undef_vnl_sample_h_
#undef vnl_sample_h_
#endif



#endif // #ifndef __itkImageNotSoRandomIteratorWithIndex_h
