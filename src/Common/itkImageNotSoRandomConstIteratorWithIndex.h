#ifndef __itkImageNotSoRandomConstIteratorWithIndex_h
#define __itkImageNotSoRandomConstIteratorWithIndex_h

/** Make sure vnl/vnl_sample.h is not included by 
 * itkImageRandomConstIteratorWithIndex.txx */
#ifndef vnl_sample_h_
	#define vnl_sample_h_
	#define __dont_forget_to_undef_vnl_sample_h_
#endif

/** The functions that replace the functions in vnl_sample.h */
#include "elx_sample.h"

#define vnl_sample_reseed elx_sample_reseed
#define vnl_sample_uniform elx_sample_uniform
#define ImageRandomConstIteratorWithIndex ImageNotSoRandomConstIteratorWithIndex

/** Make sure itkImageRandomConstIteratorWithIndex.h is really included here */
#if defined(__itkImageRandomConstIteratorWithIndex_h)
	#undef __itkImageRandomConstIteratorWithIndex_h
#else 
	#define __dont_forget_to_undef_itkconstit_h_
#endif	

#if defined(_itkImageRandomConstIteratorWithIndex_txx)
	#undef _itkImageRandomConstIteratorWithIndex_txx
#else 
	#define __dont_forget_to_undef_itkconstit_txx_
#endif	

/** include the itk-version. */
#include "itkImageRandomConstIteratorWithIndex.h"

/** clean up. */

/* Make sure the normal itkRandomIterator still can be used */
#ifdef __dont_forget_to_undef_itkconstit_h_
	#undef __dont_forget_to_undef_itkconstit_h_
	#undef __itkImageRandomConstIteratorWithIndex_h
#endif 

#ifdef __dont_forget_to_undef_itkconstit_txx_
	#undef __dont_forget_to_undef_itkconstit_txx_
	#undef _itkImageRandomConstIteratorWithIndex_txx
#endif
 
#undef vnl_sample_reseed
#undef vnl_sample_uniform
#undef ImageRandomConstIteratorWithIndex

#ifdef __dont_forget_to_undef_vnl_sample_h_
#undef __dont_forget_to_undef_vnl_sample_h_
#undef vnl_sample_h_
#endif



#endif // #ifndef __itkImageNotSoRandomConstIteratorWithIndex_h
