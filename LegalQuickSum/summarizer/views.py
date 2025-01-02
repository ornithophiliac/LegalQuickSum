
import mimetypes  # to determine file type
import os
from django.shortcuts import render
from django.http import JsonResponse, HttpResponseBadRequest
from .utils import extract_text, summarize_text #Import both functions


def upload_document(request):
    if request.method == "POST" and 'file' in request.FILES:
        file = request.FILES['file']
        uploads_dir = os.path.join(os.path.dirname(__file__), 'uploads')
        os.makedirs(uploads_dir, exist_ok=True)
        file_path = os.path.join(uploads_dir, file.name)

        try:
            with open(file_path, "wb") as f:
                for chunk in file.chunks():
                    f.write(chunk)

            content_type, encoding = mimetypes.guess_type(file_path)
            extracted_text = extract_text(file_path, content_type)

            if extracted_text is None:
                return HttpResponseBadRequest("Could not extract text from the file.")

            # Generate the summary using the updated summarize_text function.
            summary = summarize_text(extracted_text)

            os.remove(file_path)
            # Return the summary in the JsonResponse
            return JsonResponse({"summary": summary}) # only return the summary

        except Exception as e:
            return HttpResponseBadRequest(f"An error occurred: {str(e)}")

    return render(request, "summarizer/upload.html")




