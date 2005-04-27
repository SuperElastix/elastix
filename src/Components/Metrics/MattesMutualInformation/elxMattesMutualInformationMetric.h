#ifndef __elxMattesMutualInformationMetric_H__
#define __elxMattesMutualInformationMetric_H__

/** For easy changing the pixel type of the mask images: */
#define __MaskFilePixelType char

#include "elxIncludes.h"
#include "itkMattesMutualInformationImageToImageMetricWithMask.h"

#include "elxTimer.h"
#include "itkImageFileReader.h"
#include "itkCastImageFilter.h"


namespace elastix
{
using namespace itk;

	/**
	 * \class MattesMutualInformationMetric
	 * \brief A metric based on mutual information...
	 *
	 * This metric is based on an adapted version of the
	 * itk::MattesMutualInformationImageToImageMetric. 
	 *
	 * The command line argumemts used by this class are:
	 * \commandlinearg -fMask: Optional argument with the file name of a mask for
	 * the fixed image. The mask image should contain of zeros and ones, zeros indicating 
	 * pixels that are not used for the registration. \n
	 *   example: <tt> -fMask fixedmask.mhd </tt>
	 * \commandlinearg -mMask: Optional argument with the file name of a mask for
	 * the moving image. The mask image should contain of zeros and ones, zeros indicating 
	 * pixels that are not used for the registration. \n
	 *   example: <tt> -mMask movingmask.mhd </tt>
	 *
	 * The parameters used in this class are:
	 * \parameter Metric: Select this metric as follows:\n
	 * <tt> (Metric MattesMutualInformation) </tt>
	 * \parameter NumberOfHistogramBins: The size of the histogram. Must be given for each 
	 * resolution. \n
	 *   example: <tt> (NumberOfHistogramBins 32 32 64)</tt>
	 * \parameter NumberOfSpatialSamples: The number of image voxels used for computing the
	 * metric value and its derivative in each iteration. Must be given for each resolution.\n
	 *  example: <tt> (NumberOfSpatialSamples 2048 2048 4000) </tt>
	 * \parameter NumberOfResolutions: The number of resolutions.\n
	 *   example: <tt> (NumberOfResolutions 3) </tt>
	 * \parameter	UseAllPixels: Flag to force the metric to use ALL voxels for 
	 * computing the metric value and its derivative in each iteration. Must be given for each
	 * resolution. Can have values "true" or "false".\n
	 *   example: <tt> (UseAllPixels "true" "false" "true") </tt>
	 * \parameter ShowExactMetricValue: Flag that can set to "true" or "false". If "true" the 
	 * metric computes the exact metric value (computed on all voxels rather than on the set of
	 * spatial samples) and shows it each iteration. Must be given for each resolution.\n
	 * NB: If the UseallPixels flag is set to "true", this option is ignored.\n
	 *   example: <tt> (ShowAllPixels "true" "true" "false") </tt>
	 *
   * \sa MattesMutualInformationImageToImageMetricWithMask
	 * \ingroup Metrics
	 */
	
	template <class TElastix >	
		class MattesMutualInformationMetric :
		public
			MattesMutualInformationImageToImageMetricWithMask<
				ITK_TYPENAME MetricBase<TElastix>::FixedImageType,
				ITK_TYPENAME MetricBase<TElastix>::MovingImageType >,
		public MetricBase<TElastix>
	{
	public:

		/** Standard ITK-stuff.*/
		typedef MattesMutualInformationMetric									Self;
		typedef MattesMutualInformationImageToImageMetricWithMask<
			typename MetricBase<TElastix>::FixedImageType,
			typename MetricBase<TElastix>::MovingImageType >		Superclass1;
		typedef MetricBase<TElastix>													Superclass2;
		typedef SmartPointer<Self>														Pointer;
		typedef SmartPointer<const Self>											ConstPointer;
		
		/** Method for creation through the object factory. */
		itkNewMacro( Self );
		
		/** Run-time type information (and related methods). */
		itkTypeMacro( MattesMutualInformationMetric,
			MattesMutualInformationImageToImageMetricWithMask );
		
		/** Name of this class.*/
		elxClassNameMacro( "MattesMutualInformation" );

		/** Typedefs inherited from the superclass.*/
		typedef typename Superclass1::TransformType							TransformType;
		typedef typename Superclass1::TransformPointer 					TransformPointer;
		typedef typename Superclass1::TransformJacobianType			TransformJacobianType;
		typedef typename Superclass1::InterpolatorType 					InterpolatorType;
		typedef typename Superclass1::MeasureType								MeasureType;
		typedef typename Superclass1::DerivativeType 						DerivativeType;
		typedef typename Superclass1::ParametersType 						ParametersType;
		typedef typename Superclass1::FixedImageType 						FixedImageType;
		typedef typename Superclass1::MovingImageType						MovingImageType;
		typedef typename Superclass1::FixedImageConstPointer 		FixedImageConstPointer;
		typedef typename Superclass1::MovingImageConstPointer		MovingImageCosntPointer;
		typedef typename Superclass1::FixedImageIndexType				FixedImageIndexType;
		typedef typename Superclass1::FixedImageIndexValueType 	FixedImageIndexValueType;
		typedef typename Superclass1::MovingImageIndexType 			MovingImageIndexType;
		typedef typename Superclass1::FixedImagePointType				FixedImagePointType;
		typedef typename Superclass1::MovingImagePointType 			MovingImagePointType;
		
		/** The moving image dimension. */
		itkStaticConstMacro( MovingImageDimension, unsigned int,
			MovingImageType::ImageDimension );
		
		/** Other typedef's.*/
		typedef typename Superclass1::MaskPixelType							MaskPixelType;
		typedef typename Superclass1::FixedCoordRepType					FixedCoordRepType;
		typedef typename Superclass1::MovingCoordRepType 				MovingCoordRepType;
		typedef typename Superclass1::FixedMaskImageType 				FixedMaskImageType;
		typedef typename Superclass1::MovingMaskImageType				MovingMaskImageType;
		typedef typename Superclass1::FixedMaskImagePointer			FixedMaskImagePointer;
		typedef typename Superclass1::MovingMaskImagePointer		MovingMaskImagePointer;
		
		/** Typedef's inherited from Elastix.*/
		typedef typename Superclass2::ElastixType						ElastixType;
		typedef typename Superclass2::ElastixPointer				ElastixPointer;
		typedef typename Superclass2::ConfigurationType			ConfigurationType;
		typedef typename Superclass2::ConfigurationPointer	ConfigurationPointer;
		typedef typename Superclass2::RegistrationType			RegistrationType;
		typedef typename Superclass2::RegistrationPointer		RegistrationPointer;
		typedef typename Superclass2::ITKBaseType						ITKBaseType;
			
		/** Typedef for the suppport of masks.*/
		typedef __MaskFilePixelType	MaskFilePixelType; //defined at the top of this file
		/** Typedef's for the suppport of masks.*/
		typedef FixedCoordRepType		MaskCoordinateType;
		
		/** Typedef for the suppport of masks.*/
		typedef MaskImage<
			MaskFilePixelType,
			itkGetStaticConstMacro(MovingImageDimension),
			FixedCoordRepType >				FixedMaskFileImageType;
		/** Typedef for the suppport of masks.*/
		typedef MaskImage<
			MaskFilePixelType,
			itkGetStaticConstMacro(MovingImageDimension),
			MovingCoordRepType >			MovingMaskFileImageType;

		/** Typedef for the suppport of masks.*/
		typedef ImageFileReader<
			FixedMaskFileImageType > 	FixedMaskImageReaderType;

		/** Typedef for the suppport of masks.*/
		typedef ImageFileReader<
			MovingMaskFileImageType >	MovingMaskImageReaderType;
		
		/** Typedef for the suppport of masks.*/
		typedef typename FixedMaskImageReaderType::Pointer		FixedMaskImageReaderPointer;
		/** Typedef for the suppport of masks.*/
		typedef typename MovingMaskImageReaderType::Pointer		MovingMaskImageReaderPointer;

		/** Typedef for timer.*/
		typedef tmr::Timer					TimerType;
		/** Typedef for timer.*/
		typedef TimerType::Pointer	TimerPointer;
		
		/** Method that takes care of setting the parameters and showing information.*/
		virtual int BeforeAll(void);
		/** Method that takes care of setting the parameters and showing information.*/
		virtual void BeforeRegistration(void);
		/** Method that takes care of setting the parameters and showing information.*/
		virtual void BeforeEachResolution(void);
		/** Method that takes care of setting the parameters and showing information.*/
		virtual void AfterEachIteration(void);

		/** Sets up a timer to measure the intialisation time and calls the Superclass'
		 * implementation */
		virtual void Initialize(void) throw (ExceptionObject);

		/** Select a new sample set on request */
		virtual void SelectNewSamples(void);
		
	protected:

		MattesMutualInformationMetric(); 
		virtual ~MattesMutualInformationMetric() {}

		/** Declaration of member variable, for mask support.*/
		FixedMaskImageReaderPointer		m_FixedMaskImageReader;
		/** Declaration of member variable, for mask support.*/
		MovingMaskImageReaderPointer	m_MovingMaskImageReader;

		/** Flag */
		bool m_ShowExactMetricValue;
				
	private:

		MattesMutualInformationMetric( const Self& );	// purposely not implemented
		void operator=( const Self& );								// purposely not implemented
		
	}; // end class MattesMutualInformationMetric


} // end namespace elastix


#ifndef ITK_MANUAL_INSTANTIATION
#include "elxMattesMutualInformationMetric.hxx"
#endif

#endif // end #ifndef __elxMattesMutualInformationMetric_H__
