#ifndef __itkMaskImage_h
#define __itkMaskImage_h

#include "itkImage.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkBinaryBallStructuringElement.h"
#include "itkGrayscaleErodeImageFilter.h"
#include "itkGrayscaleDilateImageFilter.h"

namespace itk
{

	/**
	 * \class MaskImage
	 * \brief A class to provide mask functionality for the fixed
	 * and moving image.
	 *
	 * The MaskImage class ....
	 *
	 * \ingroup Miscellaneous
	 */

	template < class MaskPixelType = char,
		unsigned int MaskDimension = 2 , class CoordType = float >
		class MaskImage :
		public Image< MaskPixelType, MaskDimension >
	{
	public:
	
		/** Standard class typedefs.*/
		typedef MaskImage															Self;
		typedef Image<MaskPixelType, MaskDimension>		Superclass;
		typedef SmartPointer<Self>										Pointer;
		typedef SmartPointer<const Self>							ConstPointer;
		
		/** Method for creation through the object factory. */
		itkNewMacro( Self );
		
		/** Run-time type information (and related methods). */
		itkTypeMacro( MaskImage, Image );
		
		/** Get the Dimension.*/
		itkStaticConstMacro( ImageDimension, unsigned int, MaskDimension );
		
		/** Rewriting.*/
		typedef CoordType CoordinateType;
		
		/** Typedefs inherited from the Superclass.*/		
		typedef typename Superclass::PixelType							PixelType;
		typedef typename Superclass::ValueType							ValueType;
		typedef typename Superclass::InternalPixelType			InternalPixelType;
		typedef typename Superclass::AccessorType						AccessorType;
		typedef typename Superclass::PixelContainer					PixelContainer;
		typedef typename Superclass::PixelContainerPointer	PixelContainerPointer;
		typedef typename Superclass::IndexType							IndexType;
		typedef typename Superclass::OffsetType							OffsetType;
		typedef typename Superclass::SizeType								SizeType;
		typedef typename Superclass::RegionType							RegionType;
		
		/** Typedefs needed for interaction with the interpolator.*/		
		typedef NearestNeighborInterpolateImageFunction<
			Superclass, CoordinateType>											MaskInterpolator;
		typedef typename MaskInterpolator::PointType			PointType;
		typedef typename MaskInterpolator::Pointer				MaskInterpolatorPointer;
		
		/** Typedefs for interaction with the Morphological operators.*/		
		typedef BinaryBallStructuringElement<
			PixelType,
			itkGetStaticConstMacro( ImageDimension ) >			StructuringElement;
		typedef typename StructuringElement::RadiusType		RadiusType;
		typedef typename RadiusType::SizeValueType				RadiusValueType;
		typedef GrayscaleErodeImageFilter<
			Superclass, Superclass, StructuringElement >		ErodeFilter;
		typedef GrayscaleDilateImageFilter<
			Superclass, Superclass, StructuringElement >		DilateFilter;
		
		/** Evaluate if a point falls within the mask or not. */
		virtual bool IsInMask( const PointType &point ) const  
		{
			return ( (m_Interpolator->Evaluate( point )) != 0 );
			
			//May be slightly faster, but less accurate:
			//IndexType index;
			//this->TransformPhysicalPointToIndex(point, index);
			//return (this->GetPixel(index)==1);
			//assumes that the point IsInsideBuffer!
		};
				
		/** Morphological operations; the (const unsigned long)
		 * radius of the structuring element is needed as an input.
		 */
		virtual Pointer Erode( const RadiusValueType radius );	
		virtual Pointer Dilate( const RadiusValueType radius );	
		
	protected:
		
		MaskImage(); 
		virtual ~MaskImage();

		//void PrintSelf(std::ostream& os, Indent indent) const;

		/** Declarations of member variables.*/
		MaskInterpolatorPointer		m_Interpolator;
		StructuringElement				m_Ball;
		
	private:
		
		MaskImage( const Self& );				// purposely not implemented
		void operator=( const Self& );	// purposely not implemented
		
	}; // end class MaskImage


} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMaskImage.hxx"
#endif

#endif // end #ifndef __itkMaskImage_h

