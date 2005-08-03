/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile$
  Language:  C++
  Date:      $Date$
  Version:   $Revision$

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkRigidRegularizationDerivativeImageFilter_h
#define __itkRigidRegularizationDerivativeImageFilter_h

#include "itkImageToImageFilter.h"
#include "itkImage.h"

//#include "itkSecondOrderRegularizationNonSeparableOperator.h"
#include "itkNeighborhoodOperatorImageFilter.h"
#include "itkNeighborhood.h"
#include "itkNeighborhoodIterator.h"
//#include "itkBSplineKernelFunction.h"
#include "itkImageRegionIterator.h"

namespace itk
{
/**
 * \class RigidRegularizationDerivativeImageFilter
 *
 * \brief This filter computes the derivative of a penalty term on a 
 * vector-valued image. This penalty term is an isotropic measure of the
 * 1st and 2nd order spatial derivatives of an image.
 *
 * The intended use for this filter is to filter a B-spline coefficient
 * image in order to introduce a regularization term on a B-spline transform. 
 * 
 * \par
 * The RigidRegularizationDerivativeImageFilter at each pixel location is computed by
 * convolution with the itk::SecondOrderRegularizationNonSeparableOperator
 * and with other separable 1D kernels.
 *
 * \par Inputs and Outputs
 * The input to this filter is a vector-valued itk::Image of arbitrary
 * dimension. The output is a vector-valued itk::Image of the same dimension.
 *
 * \sa Image
 * \sa Neighborhood
 * \sa NeighborhoodOperator
 * \sa NeighborhoodIterator
 * \sa SecondOrderRegularizationNonSeparableOperator
 *
 * \ingroup ImageFeatureExtraction
 */

	template < class TInputImage, class TOutputImage >
	class ITK_EXPORT RigidRegularizationDerivativeImageFilter : 
		public ImageToImageFilter< TInputImage, TOutputImage > 
	{
	public:

		/** Standard itk stuff. */
		typedef RigidRegularizationDerivativeImageFilter					Self;
		typedef ImageToImageFilter< TInputImage, TOutputImage >		Superclass;
		typedef SmartPointer< Self >															Pointer;
		typedef SmartPointer< const Self >												ConstPointer;

		/** Run-time type information (and related methods). */
		itkTypeMacro( RigidRegularizationDerivativeImageFilter, ImageToImageFilter );

		/** Method for creation through the object factory. */
		itkNewMacro( Self );

		/** Extract some information from the image types. Dimensionality
		 * of the two images is assumed to be the same and of vector type.
		 * The vector dimension is assumed to be the same as the image dimension.
		 */
		typedef typename Superclass::InputImageType						InputVectorImageType;
		typedef typename Superclass::OutputImageType					OutputVectorImageType;
		typedef typename InputVectorImageType::Pointer				InputVectorImagePointer;
		typedef typename OutputVectorImageType::Pointer				OutputVectorImagePointer;
		typedef typename InputVectorImageType::PixelType			InputVectorPixelType;
		typedef typename InputVectorImageType::PixelType			OutputVectorPixelType;
		typedef typename InputVectorPixelType::ValueType			InputVectorValueType;
		typedef typename OutputVectorPixelType::ValueType			OutputVectorValueType;
		
		/** Define the dimension. */
		itkStaticConstMacro( ImageDimension, unsigned int, OutputVectorImageType::ImageDimension );

		/** Define scalar versions of the vector images. */
		typedef Image< InputVectorValueType,
			itkGetStaticConstMacro( ImageDimension ) >					InputScalarImageType;
		typedef Image< OutputVectorValueType,
			itkGetStaticConstMacro( ImageDimension ) >					OutputScalarImageType;
		typedef typename InputScalarImageType::Pointer				InputScalarImagePointer;
		typedef typename OutputScalarImageType::Pointer				OutputScalarImagePointer;

		/** Typedef support for neigborhoods, filters, etc. */
		typedef NeighborhoodOperatorImageFilter<
			InputScalarImageType, InputScalarImageType >				NOIFType;
		typedef Neighborhood< InputVectorValueType,
			itkGetStaticConstMacro( ImageDimension ) >					NeighborhoodType;
		typedef NeighborhoodIterator<
			InputScalarImageType >															NeighborhoodIteratorInputType;
		typedef typename NeighborhoodIteratorInputType
			::RadiusType																				RadiusInputType;
		typedef NeighborhoodIterator<
			OutputScalarImageType >															NeighborhoodIteratorOutputType;
		typedef typename NeighborhoodIteratorOutputType
			::RadiusType																				RadiusOutputType;
		typedef	typename NeighborhoodType::SizeType						SizeType;
		//typedef SecondOrderRegularizationNonSeparableOperator<
			//OutputVectorValueType,
			//itkGetStaticConstMacro( ImageDimension ) >					SOOperatorType;

		/** Typedef support for B-spline kernel functions. *
		typedef BSplineKernelFunction< 1 >	BSplineKernelFunctionOrder1Type;
		typedef BSplineKernelFunction< 2 >	BSplineKernelFunctionOrder2Type;
		typedef BSplineKernelFunction< 3 >	BSplineKernelFunctionOrder3Type;

		/** Typedef support for iterators. */
		typedef typename NeighborhoodType::Iterator						NIType;
		typedef ImageRegionIterator< InputScalarImageType >		InputScalarImageIteratorType;
		typedef ImageRegionIterator< OutputScalarImageType >	OutputScalarImageIteratorType;

		/** RigidRegularizationImageFilter needs a larger
		* input requested region than the output requested region (larger
		* in the direction of the derivative). As such,
		* SecondOrderRegularizationNonSeparableImageFilter needs to provide an
		* implementation for GenerateInputRequestedRegion() in order to
		* inform the pipeline execution model.
		*
		* \sa ImageToImageFilter::GenerateInputRequestedRegion()
		*/
		virtual void GenerateInputRequestedRegion() throw(InvalidRequestedRegionError);

		/** Use the image spacing information in calculations. Use this option if you
		 * want derivatives in physical space. Default is UseImageSpacingOn. */
		void SetUseImageSpacingOn()
		{ this->SetUseImageSpacing(true); }

		/** Ignore the image spacing. Use this option if you want derivatives in
		 * isotropic pixel space.  Default is UseImageSpacingOn. */
		void SetUseImageSpacingOff()
		{ this->SetUseImageSpacing(false); }

		/** Set/Get whether or not the filter will use the spacing of the input
		 * image in its calculations */
		itkSetMacro( UseImageSpacing, bool );
		itkGetMacro( UseImageSpacing, bool );
		itkGetMacro( ImageSpacingUsed, double * );

		/** Set/Get the weight of the second order part. */
		itkSetClampMacro( SecondOrderWeight, InputVectorValueType,
			0.0, NumericTraits<InputVectorValueType>::max() );
		itkGetMacro( SecondOrderWeight, InputVectorValueType );

		/** Set the coefficient matrix C. */
		itkSetMacro( RigidityImage, InputScalarImagePointer );

		/** Get the value of the rigid regulizer. */
		InputVectorValueType GetRigidRegulizerValue(void);

		/** Set the OutputDirectoryName. */
		itkSetStringMacro( OutputDirectoryName );

	protected:

		RigidRegularizationDerivativeImageFilter();
		virtual ~RigidRegularizationDerivativeImageFilter(){}

		/** Standard pipeline method. While this class does not implement a
		* ThreadedGenerateData(), its GenerateData() delegates all
		* calculations to an NeighborhoodOperatorImageFilter.  Since the
		* NeighborhoodOperatorImageFilter is multithreaded, this filter is
		* multithreaded by default.
		*/
		void GenerateData();
		void PrintSelf( std::ostream&, Indent ) const;

	private:

		RigidRegularizationDerivativeImageFilter( const Self& );	// purposely not implemented
		void operator=( const Self& );									// purposely not implemented

		/** Some private functions used for the filtering. */
		void Create1DOperator( NeighborhoodType & F, std::string WhichF, unsigned int WhichDimension );
		void CreateNDOperator( NeighborhoodType & F, std::string WhichF );
		InputScalarImagePointer FilterNonSeparable( const InputScalarImageType *, NeighborhoodType );
		InputScalarImagePointer FilterSeparable( const InputScalarImageType *, std::vector< NeighborhoodType > Operators );
		double CalculateSubPart( unsigned int dim, unsigned int part, std::vector<OutputVectorValueType> values );

		/** What image spacing to use. */
		void SetImageSpacingUsed( void );

		/** Some private variables to store spacing related stuff. */
		bool			m_UseImageSpacing;
		double		m_ImageSpacingUsed[ itkGetStaticConstMacro( ImageDimension ) ];

		/** A private variable to store the weighting of the second order part. */
		InputVectorValueType			m_SecondOrderWeight;

		/** A private variable to store if GenerateDate() has been called. */
		bool			m_GenerateDataCalled;

		/** A private variable to store the rigid metric value. */
		InputVectorValueType			m_RigidRegulizerValue;
		
		/** A private variable to store the coefficient image.
		 * This only stores a pointer to an externally kept image.
		 */
		InputScalarImagePointer		m_RigidityImage;

		/** Name of the output directory. */
		std::string m_OutputDirectoryName;

	}; // end class RigidRegularizationDerivativeImageFilter

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkRigidRegularizationDerivativeImageFilter.txx"
#endif

#endif // end #ifndef __itkRigidRegularizationDerivativeImageFilter_h
